"""Audit scientific artifacts without regenerating or changing them."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import rotate

from figure_io import verify_png

EXPECTED = {
    "convolution_input", "convolution_kernel", "circular_spatial", "padding_input",
    "padding_kernel", "linear_spatial", "insufficient_padding", "correlation_x",
    "correlation_y", "correlation_spatial", "operator_input", "forward_difference_spatial",
    "laplacian_spatial", "scaling_input", "zero_inserted", "rotation_input", "rotated_30", "rotated_back",
}


def audit(output):
    output = Path(output)
    figures = sorted(output.glob("*.png"))
    assert len(figures) == 9
    for path in figures:
        verify_png(path)
    metrics = json.loads((output / "metrics.json").read_text())
    assert len(metrics["measurements"]) == 40
    with (output / "measurements.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
    for row, record in zip(rows, metrics["measurements"]):
        assert row["case"] == record["case"]
        assert float(row["max_absolute_error"]) == record["max_absolute_error"]
        assert float(row["relative_l2_error"]) == record["relative_l2_error"]
        assert json.loads(row["settings"]) == record["settings"]
    assert max(r["max_absolute_error"] for r in metrics["measurements"]) < 1e-10
    with (output / "operator_responses.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 257
    with np.load(output / "reference_cases.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == EXPECTED
        for key in arrays.files:
            assert arrays[key].ndim == 2
            assert arrays[key].dtype in (np.dtype("float64"), np.dtype("complex128"))
            assert np.isfinite(arrays[key]).all()
        def same(a, b):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
        full = convolve2d(arrays["convolution_input"], arrays["convolution_kernel"])
        folded = np.zeros(arrays["convolution_input"].shape, dtype=complex)
        for (m, n), value in np.ndenumerate(full):
            folded[m % folded.shape[0], n % folded.shape[1]] += value
        same(arrays["circular_spatial"], folded)
        full = convolve2d(arrays["padding_input"], arrays["padding_kernel"])
        same(arrays["linear_spatial"], full)
        folded = np.zeros(arrays["insufficient_padding"].shape, dtype=complex)
        for (m, n), value in np.ndenumerate(full):
            folded[m % folded.shape[0], n % folded.shape[1]] += value
        same(arrays["insufficient_padding"], folded)
        x, y = arrays["correlation_x"], arrays["correlation_y"]
        same(arrays["correlation_spatial"], np.fft.ifft2(np.fft.fft2(x).conj()*np.fft.fft2(y)))
        image = arrays["operator_input"]
        # Explicit scalar indexing keeps the operator reference independent.
        difference, laplacian = np.empty_like(image), np.empty_like(image)
        for m, n in np.ndindex(image.shape):
            difference[m,n] = (image[m,(n+1) % image.shape[1]]-image[m,n])/1.3
            laplacian[m,n] = ((image[(m+1) % image.shape[0],n]-2*image[m,n]+image[m-1,n])/.7**2
                              +(image[m,(n+1) % image.shape[1]]-2*image[m,n]+image[m,n-1])/1.3**2)
        same(arrays["forward_difference_spatial"], difference)
        same(arrays["laplacian_spatial"], laplacian)
        inserted = np.zeros((48, 96), complex)
        inserted[::2, ::3] = arrays["scaling_input"]
        same(arrays["zero_inserted"], inserted)
        settings = dict(reshape=False, order=1, mode="constant", cval=0, prefilter=False)
        same(arrays["rotated_30"], rotate(arrays["rotation_input"], 30, **settings))
        same(arrays["rotated_back"], rotate(arrays["rotated_30"], -30, **settings))
    return dict(figures_verified=len(figures), measurements_verified=len(rows), arrays_verified=18,
                operator_frequency_rows=257, max_absolute_error=max(r["max_absolute_error"] for r in metrics["measurements"]),
                status="passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()
    print(json.dumps(audit(args.output_dir), indent=2))
