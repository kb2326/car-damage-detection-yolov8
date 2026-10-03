import threading
from pathlib import Path
from uuid import UUID, uuid4

from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.web.runner import ClaimRunner
from claimlens.web.services import pipeline_processor
from claimlens.web.workers import Worker
from tests.unit.web.helpers import deps, image


def test_an_inline_worker_runs_at_once() -> None:
    worker = Worker("t", inline=True)
    assert worker.call(lambda: 41 + 1) == 42


def test_an_inline_worker_passes_errors_on() -> None:
    def boom() -> int:
        raise KeyError("x")

    worker = Worker("t", inline=True)
    future = worker.submit(boom)
    assert isinstance(future.exception(), KeyError)


def test_a_threaded_worker_keeps_one_thread() -> None:
    worker = Worker("t")
    names = {worker.call(lambda: threading.current_thread().name) for _ in range(3)}
    worker.shutdown()
    assert len(names) == 1
    assert threading.current_thread().name not in names


def test_the_runner_processes_a_claim_on_its_worker(tmp_path: Path) -> None:
    db = tmp_path / "events.db"
    store = SQLiteEventStore(db)
    d = deps(tmp_path, store)
    claim = submit_claim(
        store, d.blobs, policy_id="P-1001", description="x", photo_paths=[image(tmp_path / "a.png")]
    )
    store.close()
    process = pipeline_processor(db, lambda s: deps(tmp_path, s))
    runner = ClaimRunner(Worker("claims", inline=True), process)
    assert runner.start(claim)
    assert not runner.running(claim)
    check = SQLiteEventStore(db)
    assert "RouteDecided" in [e.type for e in check.load(claim)]
    check.close()


def test_a_failure_is_kept_as_the_claims_error() -> None:
    def boom(claim_id: UUID) -> None:
        raise RuntimeError("detector crashed")

    runner = ClaimRunner(Worker("claims", inline=True), boom)
    claim = uuid4()
    runner.start(claim)
    assert runner.error(claim) == "RuntimeError: detector crashed"
    assert not runner.running(claim)


def test_a_second_start_while_running_is_refused() -> None:
    gate = threading.Event()

    def slow(claim_id: UUID) -> None:
        gate.wait(5)

    worker = Worker("claims")
    runner = ClaimRunner(worker, slow)
    claim = uuid4()
    assert runner.start(claim)
    assert runner.running(claim)
    assert not runner.start(claim)
    gate.set()
    worker.call(lambda: None)  # wait for the queued run to finish
    assert not runner.running(claim)
    worker.shutdown()
