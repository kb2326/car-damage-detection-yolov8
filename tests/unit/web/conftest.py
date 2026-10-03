from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claimlens.blobs import BlobStore
from claimlens.web.app import create_app
from claimlens.web.runner import ClaimRunner
from claimlens.web.services import WebServices, pipeline_processor
from claimlens.web.settings import WebSettings
from claimlens.web.workers import Worker
from tests.unit.web.helpers import deps


@pytest.fixture
def services(tmp_path: Path) -> Iterator[WebServices]:
    """The pipeline runs inline (it opens its own store per claim); the intake chat runs on a
    real dedicated thread, as in production, because its LLM gateway holds SQLite connections."""
    db = tmp_path / "events.db"
    process = pipeline_processor(db, lambda store: deps(tmp_path, store))
    intake_worker = Worker("intake")
    yield WebServices(
        db_path=db,
        blobs=BlobStore(tmp_path / "blobs"),  # the same folder deps() uses
        settings=WebSettings(max_upload_mb=1, allowed_hosts=("testserver", "127.0.0.1")),
        runner=ClaimRunner(Worker("claims", inline=True), process),
        intake_worker=intake_worker,
        intake=None,
        upload_dir=tmp_path / "uploads",
    )
    intake_worker.shutdown()


@pytest.fixture
def client(services: WebServices) -> Iterator[TestClient]:
    with TestClient(create_app(services)) as test_client:
        yield test_client
