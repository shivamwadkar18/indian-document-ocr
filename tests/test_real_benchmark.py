"""Real-crop benchmark tooling, on a fabricated detector-format split.

No real documents: images are generated arrays and every reference string is
a fictional placeholder.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from idocr.evaluation.real_benchmark import (
    CMD_QUIT,
    CMD_SKIP,
    CMD_UNCLEAR,
    FIELDS,
    MANIFEST_FILENAME,
    BenchmarkError,
    ReferenceStore,
    aggregate_report,
    alpha_rate,
    collect_candidates,
    create_benchmark,
    display_copy,
    ensure_private_dir,
    evaluate_benchmark,
    format_report,
    load_crop,
    load_manifest,
    next_sample,
    order_candidates,
    progress,
    text_metrics,
    transcribe,
    validate_reference,
)
from idocr.types import RecognitionResult
from idocr.utils.paths import find_project_root

W, H = 120, 80
# One box per field (class id = index in FIELDS), two rows of three.
BOXES = {cid: ((cid % 3) * 40 + 20, (cid // 3) * 40 + 20, 30, 16) for cid in range(6)}  # cx, cy, w, h px


def _label(cid: int, cx: float, cy: float, w: float, h: float) -> str:
    return f"{cid} {cx / W:.6f} {cy / H:.6f} {w / W:.6f} {h / H:.6f}"


@pytest.fixture
def split(tmp_path: Path) -> Path:
    """8 documents; doc 0 has a second name box; plus problem cases."""
    images, labels = tmp_path / "test" / "images", tmp_path / "test" / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    rng = np.random.default_rng(0)
    for d in range(8):
        Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)).save(images / f"doc{d}.jpg", quality=95)
        lines = [_label(cid, *BOXES[cid]) for cid in range(6)]
        if d == 0:
            lines.append(_label(0, 60, 70, 30, 10))  # second name box in the same document
        (labels / f"doc{d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Problem cases, all excluded and counted.
    (labels / "orphan.txt").write_text(_label(0, 20, 20, 10, 10) + "\n", encoding="utf-8")
    Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(images / "edge.jpg")
    (labels / "edge.txt").write_text(_label(1, 200, 20, 10, 10) + "\n" + "9 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(images / "rotated.jpg", exif=exif)
    (labels / "rotated.txt").write_text(_label(2, 20, 20, 10, 10) + "\n", encoding="utf-8")
    return images


@pytest.fixture
def bench(split: Path, tmp_path: Path) -> Path:
    out = tmp_path / "private" / "bench"
    create_benchmark(split, out, name="unit_bench", seed=7, target_per_field=3, project_root=tmp_path / "repo")
    return out


class Scripted:
    """Feeds reviewer answers; records what was shown."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.shown = []
        self.messages = []

    def ask(self, prompt):
        return self.answers.pop(0)

    def show(self, image, caption):
        self.shown.append((image.size, caption))

    def say(self, msg):
        self.messages.append(msg)


def run(bench_dir, answers, **kw):
    s = Scripted(answers)
    result = transcribe(bench_dir, ask=s.ask, show=s.show, say=s.say, **kw)
    return s, result


# --- storage / privacy -------------------------------------------------------


def test_private_dir_rules(tmp_path):
    root = find_project_root()
    with pytest.raises(BenchmarkError):
        ensure_private_dir(root / "benchmarks" / "x", root)
    with pytest.raises(BenchmarkError):
        ensure_private_dir(root / "src" / "x", root)
    assert ensure_private_dir(root / "data" / "benchmarks" / "x", root)
    assert ensure_private_dir(tmp_path / "drive" / "x", root)


# --- candidate selection -----------------------------------------------------


def test_collect_candidates_and_exclusions(split):
    candidates, excluded = collect_candidates(split)
    assert len(candidates) == 8 * 6 + 1
    assert excluded == {"missing_image": 1, "degenerate_box": 1, "unknown_class": 1, "exif_rotated_image": 1}
    assert {c.field for c in candidates} == set(FIELDS)
    assert len({c.sample_id for c in candidates}) == len(candidates)
    assert all(c.sample_id.startswith("rc-") and "doc" not in c.sample_id for c in candidates)


def test_sampling_is_reproducible(split):
    candidates, _ = collect_candidates(split)
    a = order_candidates(candidates, seed=7)
    b = order_candidates(list(reversed(candidates)), seed=7)
    c = order_candidates(candidates, seed=8)
    assert {f: [x.sample_id for x in q] for f, q in a.items()} == {f: [x.sample_id for x in q] for f, q in b.items()}
    assert any([x.sample_id for x in a[f]] != [x.sample_id for x in c[f]] for f in FIELDS)


def test_distinct_documents_first(split):
    candidates, _ = collect_candidates(split)
    name_queue = order_candidates(candidates, seed=3)["name"]
    docs = [c.document for c in name_queue]
    assert len(docs) == 9
    assert len(set(docs[:8])) == 8  # one per document before any repeat
    assert docs[8] == "doc0"


# --- create ------------------------------------------------------------------


def test_create_exports_exact_crops_and_is_immutable(split, bench, tmp_path):
    manifest = load_manifest(bench)
    assert manifest["seed"] == 7 and manifest["split"] == "test" and manifest["box_source"] == "ground_truth"
    assert sum(len(q) for q in manifest["queues"].values()) == 49
    assert len(list((bench / "crops").glob("*.png"))) == 49

    sid = manifest["queues"]["gender"][0]
    meta = manifest["samples"][sid]
    source = np.asarray(Image.open(split / meta["image"]).convert("RGB"))
    x1, y1, x2, y2 = meta["crop_window_px"]
    assert meta["crop_size"] == [x2 - x1, y2 - y1]
    assert np.array_equal(np.asarray(load_crop(bench, manifest, sid)), source[y1:y2, x1:x2])

    with pytest.raises(BenchmarkError, match="never overwritten"):
        create_benchmark(split, bench, name="unit_bench", seed=7, project_root=tmp_path / "repo")


def test_tampered_crop_detected(bench):
    manifest = load_manifest(bench)
    sid = manifest["queues"]["name"][0]
    Image.new("RGB", (5, 5)).save(bench / manifest["samples"][sid]["crop"])
    with pytest.raises(BenchmarkError, match="hash"):
        load_crop(bench, manifest, sid)


def test_display_copy_leaves_crop_untouched(bench):
    manifest = load_manifest(bench)
    sid = manifest["queues"]["name"][0]
    crop = load_crop(bench, manifest, sid)
    before = np.asarray(crop).copy()
    big = display_copy(crop, height=160)
    assert big.height == 160 and np.array_equal(np.asarray(crop), before)


# --- references --------------------------------------------------------------


@pytest.mark.parametrize("text, ok", [
    ("Ravi Kumar", True), ("12/03/1990", True), ("ABCPX1234Z", True), ("mIxEd-Case.", True),
    ("", False), (" Ravi", False), ("Ravi ", False), ("a\nb", False), (":skipx", False),
])
def test_validate_reference_never_rewrites(text, ok):
    assert (validate_reference(text) is None) == ok


def test_store_persists_resumes_and_latest_wins(bench):
    sid_a, sid_b = load_manifest(bench)["queues"]["name"][:2]
    store = ReferenceStore(bench)
    store.record(sid_a, "skipped")
    store.record(sid_b, "verified", "Test Name")
    store.record(sid_a, "verified", "Other Name")
    with store.path.open("a", encoding="utf-8") as f:
        f.write('{"sample_id": "rc-torn", "stat')  # crash mid-write

    reopened = ReferenceStore(bench).records()  # e.g. after a runtime restart
    assert reopened[sid_a]["status"] == "verified" and reopened[sid_a]["text"] == "Other Name"
    assert reopened[sid_b]["text"] == "Test Name"
    assert "rc-torn" not in reopened
    with pytest.raises(BenchmarkError):
        store.record(sid_a, "verified", " padded")
    with pytest.raises(BenchmarkError):
        store.record(sid_a, "skipped", "text")


# --- transcription session ---------------------------------------------------


def test_transcribe_skip_unclear_and_replacements(bench):
    manifest = load_manifest(bench)
    # Least-verified field first (ties in FIELDS order). A skipped name crop is
    # replaced by the next name candidate in the fixed queue.
    s, prog = run(bench, [CMD_SKIP, "Test Name", "Test Name", CMD_UNCLEAR, CMD_QUIT])
    records = ReferenceStore(bench).records()
    name_q, dob_q = manifest["queues"]["name"], manifest["queues"]["date_of_birth"]
    assert records[name_q[0]]["status"] == "skipped"
    assert records[name_q[1]] | {"recorded_at": None} == {
        "sample_id": name_q[1], "status": "verified", "text": "Test Name", "recorded_at": None}
    assert records[dob_q[0]]["status"] == "unclear"
    assert (prog["name"]["verified"], prog["name"]["skipped"]) == (1, 1)
    assert prog["date_of_birth"]["verified"] == 0 and prog["date_of_birth"]["unclear"] == 1
    assert "Stopped; progress is saved." in s.messages


def test_transcribe_resumes_and_completes(bench):
    run(bench, [CMD_QUIT])
    answers = []
    for _ in range(6 * 3):
        answers += ["Same Text", "Same Text"]
    s, prog = run(bench, answers)
    assert all(p["verified"] == 3 and p["complete"] for p in prog.values())
    assert next_sample(load_manifest(bench), ReferenceStore(bench).records()) is None
    assert not s.answers  # nothing more was asked once every field was complete


def test_double_entry_mismatch_and_whitespace_not_saved(bench):
    s, _ = run(bench, ["Text A", "Text B", " Text A", "Text A", "Text A", CMD_QUIT], max_items=2)
    records = ReferenceStore(bench).records()
    assert [r["text"] for r in records.values()] == ["Text A"]
    assert any("differ" in m for m in s.messages)
    assert any("whitespace" in m for m in s.messages)


def test_shown_caption_has_no_text_or_source(bench):
    s, _ = run(bench, ["Hidden Ref", "Hidden Ref", CMD_QUIT])
    for size, caption in s.shown:
        assert size[1] == 160
        assert caption.startswith("rc-") and "Hidden Ref" not in caption and "doc" not in caption.lower()


def test_exhausted_field_reported(bench):
    manifest = load_manifest(bench)
    store = ReferenceStore(bench)
    for sid in manifest["queues"]["gender"]:  # 8 candidates, all skipped
        store.record(sid, "skipped")
    prog = progress(manifest, store.records())
    assert prog["gender"]["exhausted"] and not prog["gender"]["complete"]
    assert prog["gender"]["verified"] == 0  # skipped never counts as verified


# --- metrics -----------------------------------------------------------------


def test_text_metrics_raw_and_case_insensitive():
    m = text_metrics(["Ravi kumar", "1234 5678 9012", "x"], ["RAVI KUMAR", "1234 5678 9012", "y"])
    # Raw: 8 case edits + 0 + 1 over 10 + 14 + 1 reference chars.
    assert m["cer"] == pytest.approx(9 / 25)
    assert m["exact_match"] == pytest.approx(1 / 3)
    assert m["cer_ci"] == pytest.approx(1 / 25)
    assert m["exact_match_ci"] == pytest.approx(2 / 3)
    assert text_metrics([], [])["cer"] is None


def test_raw_output_is_not_cleaned():
    m = text_metrics([" MALE"], ["MALE"])
    assert m["exact_match"] == 0 and m["cer"] == pytest.approx(1 / 4)


def test_alpha_rate():
    assert alpha_rate(["1234 5678 9012", "12O4 5678 9012", "12/03/1990", "l2/03/1990"]) == 0.5
    assert alpha_rate([]) is None


def test_aggregate_report_is_text_free():
    rows = [("aadhaar_number", "1111 2222 3333", "1111 2222 333O"), ("name", "Secret Person", "Secret Persen")]
    report = aggregate_report(rows, vocabulary="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
    blob = json.dumps(report)
    for text in ("1111", "Secret", "Persen", "333O"):
        assert text not in blob
    assert report["per_field"]["aadhaar_number"]["alpha_prediction_rate"] == 1.0
    assert report["per_field"]["name"]["references_outside_vocabulary"] == 1  # lowercase letters
    assert "alpha_prediction_rate" not in report["per_field"]["name"]
    assert report["per_field"]["gender"]["count"] == 0


# --- evaluate ----------------------------------------------------------------


class EchoRecognizer:
    """Returns one fixed string for every crop (stands in for CRNNRecognizer)."""

    def __init__(self, text):
        self.text = text

    def recognize_batch(self, crops):
        return [RecognitionResult(self.text, 0.5) for _ in crops]


def test_evaluate_scores_verified_only_and_keeps_text_private(bench, capsys):
    manifest = load_manifest(bench)
    store = ReferenceStore(bench)
    store.record(manifest["queues"]["aadhaar_number"][0], "verified", "1111 2222 3333")
    store.record(manifest["queues"]["aadhaar_number"][1], "skipped")
    store.record(manifest["queues"]["name"][0], "verified", "Fictional Name")

    report = evaluate_benchmark(bench, EchoRecognizer("1111 2222 3333"), run_name="unit_run",
                                model_info={"checkpoint_file": "best.pt"})
    assert report["overall"]["count"] == 2
    assert report["per_field"]["aadhaar_number"]["exact_match"] == 1.0
    assert report["per_field"]["aadhaar_number"]["alpha_prediction_rate"] == 0.0
    assert report["per_field"]["name"]["exact_match"] == 0.0
    assert report["review_status"]["aadhaar_number"]["skipped"] == 1

    saved = (bench / "reports" / "unit_run.json").read_text(encoding="utf-8")
    table = format_report(report)
    for public in (saved, table, capsys.readouterr().out):
        for secret in ("1111", "Fictional", "doc", "crops/", ".png", ".jpg"):
            assert secret not in public
    preds = [json.loads(l) for l in (bench / "predictions" / "unit_run.jsonl").read_text().splitlines()]
    assert len(preds) == 2 and preds[0]["prediction"] == "1111 2222 3333"

    with pytest.raises(BenchmarkError, match="already exists"):
        evaluate_benchmark(bench, EchoRecognizer("x"), run_name="unit_run")


def test_evaluate_requires_references(bench):
    with pytest.raises(BenchmarkError, match="no verified"):
        evaluate_benchmark(bench, EchoRecognizer("x"), run_name="empty")


def test_evaluate_with_real_crnn_checkpoint(bench, tmp_path):
    pytest.importorskip("torch")
    from idocr.models.recognizer import Vocabulary
    from idocr.models.recognizer.crnn import CRNN, CRNNConfig, CRNNRecognizer, save_checkpoint

    vocab = Vocabulary()
    ckpt = save_checkpoint(tmp_path / "tiny.pt", CRNN(CRNNConfig(num_classes=vocab.num_classes, hidden_size=16,
                                                                 lstm_layers=1)), vocab)
    manifest = load_manifest(bench)
    ReferenceStore(bench).record(manifest["queues"]["date_of_birth"][0], "verified", "01/02/2000")
    report = evaluate_benchmark(bench, CRNNRecognizer.from_checkpoint(ckpt), run_name="tiny")
    m = report["per_field"]["date_of_birth"]
    assert m["count"] == 1 and m["cer"] is not None and m["alpha_prediction_rate"] in (0.0, 1.0)
    assert (bench / MANIFEST_FILENAME).is_file()
