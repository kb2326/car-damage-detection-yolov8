"""The gateway: budget → cache → provider with retries → fallback → validation → charge and log."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from claimlens.llm.budget import Budget
from claimlens.llm.cache import CachedReply, ResponseCache
from claimlens.llm.config import LLMConfig
from claimlens.llm.log import CallLog, LLMCall
from claimlens.llm.provider import (
    Provider,
    ProviderFatalError,
    ProviderReply,
    ProviderTransientError,
)
from claimlens.llm.types import InvalidModelOutput, LLMRequest, LLMResponse, LLMUnavailable, Message

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _extract_json(text: str) -> str:
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def _schema_system(system: str, schema: type[BaseModel]) -> str:
    return (
        f"{system}\n\nReply with only a JSON object (no prose) that matches this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema(), sort_keys=True)}"
    ).strip()


class Gateway:
    def __init__(
        self,
        config: LLMConfig,
        provider: Provider,
        budget: Budget,
        cache: ResponseCache,
        log: CallLog,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config = config
        self._provider = provider
        self._budget = budget
        self._cache = cache
        self._log = log
        self._sleep = sleep
        self._clock = clock

    def _call(
        self, models: list[str], system: str, messages: list[Message], max_tokens: int
    ) -> tuple[str, ProviderReply, int]:
        attempts = 0
        for model in models:
            for attempt in range(self._config.limits.attempts_per_model):
                attempts += 1
                try:
                    return (
                        model,
                        self._provider.complete(model, system, messages, max_tokens),
                        attempts,
                    )
                except ProviderTransientError:
                    if attempt + 1 < self._config.limits.attempts_per_model:
                        self._sleep(self._config.limits.backoff_seconds * 2**attempt)
                except ProviderFatalError as exc:
                    raise LLMUnavailable(f"LLM request rejected: {exc}") from None
        raise _AllDownError(attempts)

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = self._clock()
        request_id = uuid.uuid4().hex
        schema = request.output_schema
        system = _schema_system(request.system, schema) if schema else request.system
        max_tokens = request.max_tokens or self._config.limits.max_tokens
        models = self._config.models_for(request.tier)
        prompt_sha = hashlib.sha256(
            json.dumps([system, [m.model_dump() for m in request.messages]]).encode()
        ).hexdigest()

        def record(
            outcome: str, model: str, tin: int, tout: int, cost: float, cached: bool, attempts: int
        ) -> None:
            self._log(
                LLMCall(
                    request_id=request_id,
                    claim_id=request.claim_id,
                    prompt_id=request.prompt_id,
                    model=model,
                    input_tokens=tin,
                    output_tokens=tout,
                    cost_usd=round(cost, 6),
                    latency_ms=int((self._clock() - started) * 1000),
                    cached=cached,
                    attempts=attempts,
                    outcome=outcome,
                    prompt_sha256=prompt_sha,
                )
            )

        key = self._cache.key(
            models[0], system, request.messages, schema.__name__ if schema else None, max_tokens
        )
        hit = self._cache.get(key)
        if hit is not None:
            hit_parsed = schema.model_validate_json(_extract_json(hit.text)) if schema else None
            record("ok", hit.model, hit.input_tokens, hit.output_tokens, 0.0, True, 0)
            return LLMResponse(
                text=hit.text,
                parsed=hit_parsed,
                model=hit.model,
                input_tokens=hit.input_tokens,
                output_tokens=hit.output_tokens,
                cost_usd=0.0,
                cached=True,
                latency_ms=int((self._clock() - started) * 1000),
                attempts=0,
            )

        self._budget.check(request.claim_id)
        messages = list(request.messages)
        try:
            model, reply, attempts = self._call(models, system, messages, max_tokens)
        except _AllDownError as down:
            record("unavailable", models[-1], 0, 0, 0.0, False, down.attempts)
            raise LLMUnavailable("all LLM models are unavailable; route to a person") from None
        tin, tout = reply.input_tokens, reply.output_tokens
        cost = self._config.cost(model, tin, tout)
        text = reply.text
        parsed: BaseModel | None = None
        if schema is not None:
            try:
                parsed = schema.model_validate_json(_extract_json(text))
            except ValidationError as first_error:
                repair = [
                    *messages,
                    Message(role="assistant", content=text),
                    Message(
                        role="user",
                        content=(
                            f"Your reply was not valid: {first_error.errors()[:3]}. "
                            "Reply again with only the JSON object."
                        ),
                    ),
                ]
                try:
                    model, reply, more = self._call([model], system, repair, max_tokens)
                except _AllDownError as down:
                    self._budget.charge(request.claim_id, cost)
                    record("unavailable", model, tin, tout, cost, False, attempts + down.attempts)
                    raise LLMUnavailable(
                        "LLM unavailable during repair; route to a person"
                    ) from None
                attempts += more
                tin, tout = tin + reply.input_tokens, tout + reply.output_tokens
                cost += self._config.cost(model, reply.input_tokens, reply.output_tokens)
                text = reply.text
                try:
                    parsed = schema.model_validate_json(_extract_json(text))
                except ValidationError:
                    self._budget.charge(request.claim_id, cost)
                    record("invalid_output", model, tin, tout, cost, False, attempts)
                    raise InvalidModelOutput("model output failed validation twice") from None
        self._budget.charge(request.claim_id, cost)
        self._cache.put(
            key, CachedReply(text=text, model=model, input_tokens=tin, output_tokens=tout)
        )
        record("ok", model, tin, tout, cost, False, attempts)
        return LLMResponse(
            text=text,
            parsed=parsed,
            model=model,
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=cost,
            cached=False,
            latency_ms=int((self._clock() - started) * 1000),
            attempts=attempts,
        )


class _AllDownError(Exception):
    def __init__(self, attempts: int) -> None:
        super().__init__(attempts)
        self.attempts = attempts
