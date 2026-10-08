import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from idocr.models.recognizer import BLANK_INDEX, DEFAULT_ALPHABET, TextRecognizer, Vocabulary  # noqa: E402
from idocr.models.recognizer.crnn import (  # noqa: E402
    CRNN,
    CRNNConfig,
    CRNNRecognizer,
    greedy_decode,
    images_to_tensor,
    load_checkpoint,
    model_from_checkpoint,
    postprocess_result,
    save_checkpoint,
)
from idocr.types import RecognitionResult  # noqa: E402

FIELD_TEXTS = ["RAHUL JOSHI", "SURESH PANDEY", "16/09/1973", "FEMALE", "7686 9458 2719", "ABCPJ1234F"]


# --- vocabulary ------------------------------------------------------------


def test_alphabet_and_blank() -> None:
    vocab = Vocabulary()
    assert vocab.alphabet == DEFAULT_ALPHABET
    assert set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/ ") == set(vocab.alphabet)
    assert vocab.num_classes == 39
    assert BLANK_INDEX == 0
    assert BLANK_INDEX not in vocab.encode(DEFAULT_ALPHABET)


@pytest.mark.parametrize("text", FIELD_TEXTS)
def test_encode_decode_roundtrip(text: str) -> None:
    vocab = Vocabulary()
    assert vocab.decode(vocab.encode(text)) == text


def test_encode_rejects_unknown_and_empty() -> None:
    vocab = Vocabulary()
    with pytest.raises(ValueError):
        vocab.encode("rahul")
    with pytest.raises(ValueError):
        vocab.encode("A-B")
    with pytest.raises(ValueError):
        vocab.encode("")


def test_duplicate_alphabet_rejected() -> None:
    with pytest.raises(ValueError):
        Vocabulary("AAB")


def test_ctc_decode_collapses_repeats_and_blanks() -> None:
    vocab = Vocabulary()
    a, b = vocab.encode("AB")
    assert vocab.ctc_decode([0, a, a, 0, b, b, 0]) == "AB"
    # A blank between repeats keeps both characters ("AA" in "RAAJ").
    assert vocab.ctc_decode([a, 0, a]) == "AA"
    assert vocab.ctc_decode([a, a, a]) == "A"
    assert vocab.ctc_decode([0, 0]) == ""


# --- model -----------------------------------------------------------------


def _line(width: int, value: int = 255) -> Image.Image:
    return Image.new("L", (width, 56), value)


def test_images_to_tensor_pads_white() -> None:
    batch, widths = images_to_tensor([_line(100, 0), _line(157, 0)])
    assert batch.shape == (2, 1, 56, 180)  # 157 + 20 white margin, rounded up to a multiple of 4
    assert widths.tolist() == [100, 157]
    assert batch.min() == -1.0
    assert torch.all(batch[0, :, :, 100:] == 1.0)  # white padding


def test_images_to_tensor_rejects_mixed_heights() -> None:
    with pytest.raises(ValueError):
        images_to_tensor([_line(100), Image.new("L", (100, 40), 255)])


def test_forward_shapes() -> None:
    model = CRNN().eval()
    batch, widths = images_to_tensor([_line(160), _line(400)])
    out = model(batch, widths)

    assert out.shape == (105, 2, 39)  # (400 + 20 margin) // 4 frames; valid: 400 // 4
    assert out.dtype == torch.float32
    assert torch.allclose(out.exp().sum(-1), torch.ones(105, 2), atol=1e-4)
    assert CRNN.output_lengths(widths).tolist() == [40, 100]


def test_padding_does_not_change_prediction() -> None:
    torch.manual_seed(0)
    model = CRNN().eval()
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (56, 160), dtype=np.uint8))
    alone, w1 = images_to_tensor([image])
    padded, w2 = images_to_tensor([image, _line(400)])
    with torch.no_grad():
        a = model(alone, w1)[:40, 0]
        b = model(padded, w2)[:40, 0]
    assert torch.allclose(a, b, atol=1e-4)


def _toy_batch(vocab: Vocabulary):
    batch, widths = images_to_tensor([_line(240, 0), _line(320, 0)])
    texts = ["MALE", "16/09/1973"]
    targets = [vocab.encode(t) for t in texts]
    return (
        batch,
        widths,
        torch.tensor([i for t in targets for i in t]),
        torch.tensor([len(t) for t in targets]),
    )


def test_ctc_loss_and_backward_step() -> None:
    torch.manual_seed(0)
    vocab = Vocabulary()
    model = CRNN()
    images, widths, targets, target_lengths = _toy_batch(vocab)
    criterion = torch.nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(15):
        log_probs = model(images, widths)
        loss = criterion(log_probs, targets, CRNN.output_lengths(widths), target_lengths)
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_greedy_decode_from_log_probs() -> None:
    vocab = Vocabulary()
    m, a = vocab.encode("MA")
    frames = [m, m, 0, a, 0]
    log_probs = torch.full((5, 1, vocab.num_classes), -20.0)
    for t, c in enumerate(frames):
        log_probs[t, 0, c] = 0.0
    [(text, confidence)] = greedy_decode(log_probs, torch.tensor([5]), vocab)
    assert text == "MA"
    assert confidence == pytest.approx(1.0, abs=1e-6)
    [(short, _)] = greedy_decode(log_probs, torch.tensor([2]), vocab)
    assert short == "M"
    blank = torch.full((3, 1, vocab.num_classes), -20.0)
    blank[:, 0, 0] = 0.0
    assert greedy_decode(blank, torch.tensor([3]), vocab) == [("", 0.0)]


# --- checkpoint + recognizer interface -------------------------------------


def test_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    model = CRNN(CRNNConfig(hidden_size=32, lstm_layers=1)).eval()
    vocab = Vocabulary()
    path = save_checkpoint(tmp_path / "ckpt" / "m.pt", model, vocab, epoch=3)

    checkpoint = load_checkpoint(path)
    restored, restored_vocab = model_from_checkpoint(checkpoint)
    restored.eval()

    assert checkpoint["epoch"] == 3
    assert checkpoint["preprocessing"] == {"target_height": 48, "horizontal_padding": 8, "vertical_padding": 4}
    assert restored.config == model.config
    assert restored_vocab.alphabet == vocab.alphabet
    batch, widths = images_to_tensor([_line(200, 30)])
    with torch.no_grad():
        assert torch.equal(model(batch, widths), restored(batch, widths))


def test_recognizer_interface(tmp_path) -> None:
    torch.manual_seed(0)
    path = save_checkpoint(tmp_path / "m.pt", CRNN(), Vocabulary())
    recognizer = CRNNRecognizer.from_checkpoint(path, batch_size=2)
    assert isinstance(recognizer, TextRecognizer)

    gray = np.full((30, 200), 230, dtype=np.uint8)
    rgb = np.full((25, 120, 3), 200, dtype=np.uint8)

    single = recognizer.recognize(gray)
    batch = recognizer.recognize_batch([gray, rgb, gray])

    assert isinstance(single, RecognitionResult)
    assert len(batch) == 3
    # Batching (and padding) does not change results beyond float noise.
    assert batch[0].text == single.text
    assert batch[0].confidence == pytest.approx(single.confidence, rel=1e-4)
    for result in batch:
        assert set(result.text) <= set(DEFAULT_ALPHABET)
        assert 0.0 <= result.confidence <= 1.0


def test_recognizer_rejects_bad_input(tmp_path) -> None:
    recognizer = CRNNRecognizer.from_checkpoint(save_checkpoint(tmp_path / "m.pt", CRNN(), Vocabulary()))
    with pytest.raises(TypeError):
        recognizer.recognize(np.zeros((10, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        recognizer.recognize(np.zeros((0, 10), dtype=np.uint8))


def test_missing_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        CRNNRecognizer.from_checkpoint(tmp_path / "nope.pt")


# --- post-processing -------------------------------------------------------


@pytest.mark.parametrize(
    ("decoded", "expected"),
    [
        (" SANJAY SINGH ", "SANJAY SINGH"),
        ("3878 5427 0209", "3878 5427 0209"),
        ("31/10/1985", "31/10/1985"),
        ("  RAM  KUMAR  SINGH ", "RAM  KUMAR  SINGH"),  # internal spaces kept as-is
        (" / 31/10/1985 / ", "/ 31/10/1985 /"),
    ],
)
def test_postprocess_strips_only_outer_whitespace(decoded: str, expected: str) -> None:
    result = postprocess_result(decoded, 0.9)
    assert result.text == expected
    assert result.confidence == 0.9


def test_postprocess_whitespace_only_is_empty() -> None:
    assert postprocess_result("   ", 0.9) == RecognitionResult(text="", confidence=0.0)


def test_recognize_and_batch_apply_postprocessing(tmp_path, monkeypatch) -> None:
    import idocr.models.recognizer.crnn as crnn

    decoded = [" SANJAY SINGH ", "3878 5427 0209", "31/10/1985 ", "  "]
    monkeypatch.setattr(
        crnn, "greedy_decode",
        lambda log_probs, lengths, vocab: [(t, 0.8) for t in decoded[: log_probs.size(1)]],
    )
    recognizer = CRNNRecognizer.from_checkpoint(save_checkpoint(tmp_path / "m.pt", CRNN(), Vocabulary()))
    crop = np.full((30, 200), 230, dtype=np.uint8)

    assert recognizer.recognize(crop).text == "SANJAY SINGH"
    batch = recognizer.recognize_batch([crop] * 4)
    assert [r.text for r in batch] == ["SANJAY SINGH", "3878 5427 0209", "31/10/1985", ""]
    assert [r.confidence for r in batch] == [0.8, 0.8, 0.8, 0.0]
