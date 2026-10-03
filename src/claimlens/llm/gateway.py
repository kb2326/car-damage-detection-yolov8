"""The gateway: cache → budget → provider with retries → fallback → validation → charge and log.

Every outcome is charged for what it cost and leaves one call record: ok, invalid_output,
unavailable (all models down) or rejected (the provider refused the request).
"""

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
from claimlens.llm.types import (
    InvalidModelOutput,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMUnavailable,
    Message,
    ToolSpec,
)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_CHARS_PER_TOKEN = 3  # a cautious estimate: real English text averages about 4


def _extract_json(text: str) -> str:
    """The JSON object in a reply: the whole text if it parses, else a fenced block, else braces."""
    stripped = text.strip()
    try:
        json.loads(stripped)
    except ValueError:
        pass
    else:
        return stripped
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


class _FailedError(Exception):
    """A provider call that did not produce a reply."""

    def __init__(self, outcome: str, attempts: int, detail: str) -> None:
        super().__init__(detail)
        self.outcome = outcome
        self.attempts = attempts
        self.detail = detail


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
        self,
        models: list[str],
        system: str,
        messages: list[Message],
        max_tokens: int,
        tools: tuple[ToolSpec, ...] = (),
        tool_choice: str | None = None,
    ) -> tuple[str, ProviderReply, int]:
        attempts = 0
        for model in models:
            for attempt in range(self._config.limits.attempts_per_model):
                attempts += 1
                try:
                    reply = self._provider.complete(
                        model, system, messages, max_tokens, tools, tool_choice
                    )
                except ProviderTransientError:
                    if attempt + 1 < self._config.limits.attempts_per_model:
                        self._sleep(self._config.limits.backoff_seconds * 2**attempt)
                except ProviderFatalError as exc:
                    # Another model would be refused the same way, so there is no fallback.
                    raise _FailedError(
                        "rejected", attempts, f"LLM request rejected: {exc}"
                    ) from None
                else:
                    return model, reply, attempts
        raise _FailedError(
            "unavailable", attempts, "all LLM models are unavailable; route to a person"
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = self._clock()
        request_id = uuid.uuid4().hex
        limits = self._config.limits
        schema = request.output_schema
        system = _schema_system(request.system, schema) if schema else request.system
        max_tokens = min(request.max_tokens or limits.max_tokens, limits.max_tokens_ceiling)
        try:
            models = self._config.models_for(request.tier)
        except ValueError as exc:
            raise LLMError(str(exc)) from None
        prompt_sha = hashlib.sha256(
            json.dumps(
                [
                    system,
                    [m.model_dump() for m in request.messages],
                    [t.model_dump() for t in request.tools],
                ]
            ).encode()
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
            models[0],
            system,
            request.messages,
            schema.__name__ if schema else None,
            max_tokens,
            request.tools,
            request.tool_choice,
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
                tool_calls=hit.tool_calls,
            )

        # Refuse before spending: the worst case is a full-length reply to this prompt.
        prompt_chars = (
            len(system)
            + sum(len(m.model_dump_json()) for m in request.messages)
            + sum(len(t.model_dump_json()) for t in request.tools)
        )
        worst_case = self._config.cost(models[0], prompt_chars // _CHARS_PER_TOKEN + 1, max_tokens)
        self._budget.check(request.claim_id, worst_case)

        messages = list(request.messages)
        tin = tout = attempts = 0
        cost = 0.0
        model = models[0]

        def fail(failure: _FailedError) -> LLMUnavailable:
            """Charge what was already spent, record the failure, and build the error."""
            if cost:
                self._budget.charge(request.claim_id, cost)
            record(failure.outcome, model, tin, tout, cost, False, attempts + failure.attempts)
            return LLMUnavailable(failure.detail)

        try:
            model, reply, attempts = self._call(
                models, system, messages, max_tokens, request.tools, request.tool_choice
            )
        except _FailedError as failure:
            raise fail(failure) from None
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
                    Message(role="assistant", content=text or "(empty reply)"),
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
                except _FailedError as failure:
                    raise fail(failure) from None
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
            key,
            CachedReply(
                text=text,
                model=model,
                input_tokens=tin,
                output_tokens=tout,
                tool_calls=reply.tool_calls,
            ),
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
            tool_calls=reply.tool_calls,
        )
