from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiment import synthetic_inputs
import low_pass as lp
from plot_io import save_figure, verify_png
from verify_artifacts import audit, _manifest_bytes


def test_committed_artifact_integrity_and_numerical_references():
    report = audit(Path(__file__).resolve().parents[1] / "output")
    assert report["status"] == "passed"


def test_artifact_manifest_is_portable_across_lf_and_crlf(tmp_path):
    output = Path(__file__).resolve().parents[1] / "output"
    import hashlib
    import json
    manifest = {entry["path"]: entry for entry in
                json.loads((output / "artifact_manifest.json").read_text())}
    for name in ("measurements.csv", "radial_responses.csv", "benchmarks.csv", "metrics.json"):
        expected = manifest[name]
        canonical = _manifest_bytes(output / name)
        assert len(canonical) == expected["bytes"]
        assert hashlib.sha256(canonical).hexdigest() == expected["sha256"]
        windows_file = tmp_path / name
        windows_file.write_bytes(canonical.replace(b"\n", b"\r\n"))
        assert _manifest_bytes(windows_file) == canonical
        windows_file.write_bytes(windows_file.read_bytes() + b"X")
        assert hashlib.sha256(_manifest_bytes(windows_file)).hexdigest() != expected["sha256"]


def test_actual_figure_publication_and_truncated_rejection(tmp_path):
    figure, axis = plt.subplots(figsize=(2, 2))
    axis.imshow(lp.gaussian_low_pass((5, 7), .1))
    target = tmp_path / "valid.png"
    try:
        save_figure(figure, target)
    finally:
        plt.close(figure)
    verify_png(target)
    assert target.stat().st_size > 100 and not list(tmp_path.glob("*.tmp"))
    broken = tmp_path / "truncated.png"
    broken.write_bytes(target.read_bytes()[:40])
    with pytest.raises(ValueError, match="truncated.png"):
        verify_png(broken)


def test_synthetic_reproducibility_against_fixed_analytical_inputs():
    first, clean, noisy = synthetic_inputs()
    second, clean_again, noisy_again = synthetic_inputs()
    for key in first:assert np.array_equal(first[key], second[key])
    assert np.array_equal(clean, clean_again) and np.array_equal(noisy, noisy_again)
    noise = np.random.default_rng(404).normal(0, .12, clean.shape)
    np.testing.assert_allclose(noisy, clean+noise, rtol=0, atol=0)
    n = np.arange(160)[None, :]
    expected = .5+.25*np.cos(2*np.pi*4*n/160)+.2*np.cos(2*np.pi*36*n/160)
    np.testing.assert_allclose(first["mixed"], np.broadcast_to(expected, first["mixed"].shape), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kind", ["ideal", "butterworth", "gaussian"])
@pytest.mark.parametrize("complex_input", [False, True])
def test_filter_reproducibility_and_energy_nonincrease(kind, complex_input):
    rng = np.random.default_rng(409)
    source = rng.normal(size=(31, 40))
    if complex_input:source = source+1j*rng.normal(size=source.shape)
    first = lp.filter_image(source, .1, kind).values
    second = lp.filter_image(source, .1, kind).values
    assert np.array_equal(first, second)
    assert np.sum(abs(first)**2) <= np.sum(abs(source)**2)*(1+1e-12)
