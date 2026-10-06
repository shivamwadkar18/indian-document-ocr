from pathlib import Path

import pytest

from idocr.utils import ProjectPaths, deep_merge, find_project_root, load_config, load_default_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_find_project_root():
    assert find_project_root() == REPO_ROOT
    assert find_project_root(REPO_ROOT / "src" / "idocr" / "data") == REPO_ROOT


def test_find_project_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("IDOCR_PROJECT_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        find_project_root()


def test_default_config_loads():
    cfg = load_default_config()
    assert cfg["documents"] == ["aadhaar", "pan", "driving_license"]
    assert "audit" in cfg and "paths" in cfg


def test_load_config_rejects_non_mapping(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_load_config_empty_file(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_config(p) == {}


def test_synthetic_config_loads():
    from idocr.data.synthetic.driving_license import GeneratorConfig

    cfg = GeneratorConfig.from_dict(load_config(REPO_ROOT / "configs" / "synthetic_driving_license.yaml"))
    assert cfg.num_samples == 0
    assert cfg.degradation.jpeg_quality == (60, 95)


def test_deep_merge():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}


def test_project_paths_and_directories_exist():
    paths = ProjectPaths.from_config(load_default_config())
    for d in (paths.raw, paths.processed, paths.synthetic, paths.reports, paths.models, paths.experiments):
        assert d.is_dir(), d
    for doc in ("aadhaar", "pan", "driving_license"):
        assert paths.raw_dir(doc).is_dir()
