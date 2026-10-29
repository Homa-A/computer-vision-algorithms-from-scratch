"""Read-only integrity and independent numerical audit of generated Lab 04 data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from plot_io import verify_png

EXPECTED_FIGURES = {
    "transfer_functions.png", "radial_responses.png", "synthetic_gallery.png", "cutoff_effects.png",
    "butterworth_orders.png", "edge_ringing.png", "gaussian_spatial.png", "boundary_effects.png",
    "quantitative_comparison.png", "runtime_comparison.png",
}
def _manifest_bytes(path):
    """Compare text artifact contents independent of Git's LF/CRLF checkout mode.

    PNG and NumPy binaries always retain byte-exact integrity verification.
    For CSV/JSON, only CRLF-to-LF conversion is permitted; all other edits
    still fail the recorded length and SHA-256 checks.
    """
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if path.suffix in (".csv", ".json") else data


EXPECTED_ARRAYS = {
    "input_impulse", "input_constant", "input_step", "input_rectangle", "input_mixed", "input_texture",
    "clean", "noisy", "profile_ideal", "profile_butterworth1", "profile_butterworth2", "profile_butterworth4",
    "profile_butterworth8", "profile_gaussian", "filtered_ideal", "filtered_butterworth", "filtered_gaussian",
    "mask_ideal", "mask_butterworth", "mask_gaussian", "gaussian_exact_0", "gaussian_sampled_0",
    "gaussian_exact_1", "gaussian_sampled_1", "boundary_input", "boundary_1", "boundary_2", "boundary_3",
}


def audit(output):
    output = Path(output)
    assert {p.name for p in output.glob("*.png")} == EXPECTED_FIGURES
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    expected = EXPECTED_FIGURES | {"measurements.csv", "radial_responses.csv", "benchmarks.csv", "reference_arrays.npz", "metrics.json"}
    assert {r["path"] for r in manifest} == expected and len(manifest) == len(expected)
    for entry in manifest:
        path = output / entry["path"]
        data = _manifest_bytes(path)
        assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"], str(path)
        if path.suffix == ".png":verify_png(path)
    metrics = json.loads((output / "metrics.json").read_text())
    with (output / "measurements.csv").open(newline="") as handle: rows = list(csv.DictReader(handle))
    assert len(rows) == len(metrics["measurements"]) == 18
    for row, record in zip(rows, metrics["measurements"]):
        assert row.keys() == record.keys()
        for key, value in record.items():
            if value is None:assert row[key] == ""
            elif isinstance(value, str):assert row[key] == value
            else:assert float(row[key]) == value
    with (output / "radial_responses.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 6006
    with (output / "benchmarks.csv").open(newline="") as handle:
        timings = list(csv.DictReader(handle))
        assert len(timings) == 9
        for timing in timings:
            assert 0 < float(timing["minimum_seconds"]) <= float(timing["median_seconds"]) <= float(timing["maximum_seconds"])
    assert len(metrics["identity_checks"]) == 12
    assert max(r["max_absolute_error"] for r in metrics["identity_checks"]) < 1e-10
    with np.load(output / "reference_arrays.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == EXPECTED_ARRAYS
        for key in arrays.files:
            assert arrays[key].ndim == 2 and np.isfinite(arrays[key]).all()
            assert arrays[key].dtype in (np.dtype("float64"), np.dtype("complex128"))
        for kind in ("ideal", "butterworth", "gaussian"):
            for shape, array in (((129, 161), arrays["mask_"+kind]), ((128, 160), None)):
                # Radius constructed directly from modular integer coordinates.
                m, n = np.indices(shape)
                distance = np.sqrt((np.minimum(m, shape[0]-m)/shape[0])**2+(np.minimum(n, shape[1]-n)/shape[1])**2)
                response = (distance <= .08).astype(float) if kind == "ideal" else 1/(1+(distance/.08)**4) if kind == "butterworth" else np.exp(-distance**2/(2*.08**2))
                if array is not None:np.testing.assert_allclose(array, response, rtol=1e-10, atol=1e-10)
                else:np.testing.assert_allclose(arrays["filtered_"+kind], np.fft.ifft2(np.fft.fft2(arrays["noisy"])*response), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(arrays["input_constant"], .7, rtol=0, atol=0)
        assert arrays["input_impulse"].sum() == 1
        assert float(np.max(abs(arrays["filtered_gaussian"].imag))) < 1e-10
    return dict(status="passed", pngs=10, arrays=28, filter_cases=18, radial_rows=6006, benchmark_cases=9,
                maximum_identity_error=max(r["max_absolute_error"] for r in metrics["identity_checks"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()
    print(json.dumps(audit(args.output_dir), indent=2))
