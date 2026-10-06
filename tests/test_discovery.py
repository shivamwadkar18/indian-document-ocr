from idocr.data.discovery import discover_datasets, iter_files


def test_discover_datasets(tmp_path, make_image):
    (tmp_path / "aadhaar").mkdir()
    (tmp_path / "aadhaar" / ".gitkeep").touch()
    make_image(tmp_path / "pan" / "train" / "a.png")

    locs = {l.document_type: l for l in discover_datasets(tmp_path, ["aadhaar", "pan", "driving_license"])}

    assert locs["aadhaar"].exists and locs["aadhaar"].is_empty  # .gitkeep is not data
    assert locs["pan"].file_count == 1
    assert not locs["driving_license"].exists


def test_iter_files_skips_placeholders(tmp_path):
    (tmp_path / ".gitkeep").touch()
    (tmp_path / "x.bin").write_bytes(b"\x00")
    assert [p.name for p in iter_files(tmp_path)] == ["x.bin"]
