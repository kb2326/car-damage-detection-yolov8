import claimlens


def test_package_exposes_version() -> None:
    assert claimlens.__version__ == "0.1.0"
