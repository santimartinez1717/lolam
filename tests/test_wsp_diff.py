import json
import struct

import numpy as np

from lolam.wsp.diff import SafetensorsFile, analyze_delta, main


def _write_safetensors(path, tensors: dict[str, np.ndarray], bf16: set[str] = frozenset()):
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        if name in bf16:
            raw = (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
            dtype = "BF16"
        else:
            raw, dtype = arr.astype(np.float32).tobytes(), "F32"
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_bf16_roundtrip(tmp_path):
    x = np.array([[1.0, -2.5], [0.15625, 3.0]], dtype=np.float32)  # exactos en bf16
    _write_safetensors(tmp_path / "m.safetensors", {"w": x}, bf16={"w"})
    np.testing.assert_array_equal(SafetensorsFile(tmp_path / "m.safetensors").tensor("w"), x)


def test_rank1_edit_detected(tmp_path):
    rng = np.random.default_rng(0)
    w = rng.normal(size=(16, 32)).astype(np.float32)
    d = rng.normal(size=16)
    d /= np.linalg.norm(d)
    # Como la edición WSP: quitar la componente en dirección d de unas pocas columnas.
    cols = [3, 7, 11]
    w2 = w.copy()
    w2[:, cols] -= np.outer(d, d @ w[:, cols])

    info, vecs = analyze_delta(w, w2)
    assert info["rank"] == 1
    assert info["changed_cols"] == cols
    assert abs(abs(vecs["U"][:, 0] @ d) - 1) < 1e-5  # recupera la dirección

    base, edit = tmp_path / "base", tmp_path / "edit"
    base.mkdir(), edit.mkdir()
    same = rng.normal(size=(4, 4)).astype(np.float32)
    _write_safetensors(base / "model.safetensors", {"a": same, "w": w})
    _write_safetensors(edit / "model.safetensors", {"a": same, "w": w2})
    main([str(base), str(edit), "--out", str(tmp_path / "out")])
    rep = json.loads((tmp_path / "out" / "diff.json").read_text())
    assert list(rep["changed"]) == ["w"]
    assert (tmp_path / "out" / "directions.npz").exists()
