import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: compiled HMC or end-to-end runs")


@pytest.fixture(autouse=True)
def _runtime_dirs_in_tmp(monkeypatch, tmp_path_factory):
    import main as main_module

    root = tmp_path_factory.mktemp("runtime")
    monkeypatch.setattr(main_module, "_demand_design_dumps_dir", lambda: root / "dumps")
