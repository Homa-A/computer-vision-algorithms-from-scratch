# Assignment 02 - Hough Transform, Denoising, and Edge Detection

This assignment moves from image manipulation toward more explicit computer-vision operations: geometric detection, non-local denoising, and multi-stage edge detection.

## Experiments

### 1. Circular Hough Transform

`01-circular-hough/`

Implements circular Hough voting over center and radius, followed by geometric validation of candidate components. The experiment also estimates each detected circle's area and classifies whether the circular part contains an enclosed hole.

On the provided parts image:

- **16 circular parts** are detected.
- **4 parts** are classified as containing holes.
- Squares are rejected by a radial-residual validation step.

![Hough result](01-circular-hough/output/q1_summary.png)

### 2. Mean filtering vs. Non-Local Means

`02-nlm-denoising/`

The Baboon image is converted to grayscale, corrupted with zero-mean Gaussian noise of variance `5`, then denoised with:

- a `3x3` mean filter;
- a custom **Non-Local Means (NLM)** implementation.

| Image / Method | MSE | PSNR |
| --- | ---: | ---: |
| Noisy | 5.069629 | 41.081 dB |
| Mean 3x3 | 334.713013 | 22.884 dB |
| NLM | 5.025563 | 41.119 dB |

The low noise level and highly textured image make the naive mean filter destructive: it suppresses true high-frequency texture along with noise. NLM preserves detail much better and provides a small PSNR improvement in this particular setting.

![Denoising result](02-nlm-denoising/output/q2_comparison.png)

### 3. Canny, Sobel, Prewitt, and Roberts

`03-edge-detection/`

Canny is implemented as the complete sequence of:

1. Gaussian smoothing;
2. Sobel gradients;
3. non-maximum suppression;
4. double thresholding;
5. hysteresis edge tracking.

The assignment also implements Sobel, Prewitt, and Roberts operators directly and compares thresholded outputs on Camera256.

A Canny threshold sweep shows the expected reduction in detected edge pixels as thresholds increase:

| Canny thresholds | Edge pixels |
| --- | ---: |
| 5 / 12 | 7459 |
| 10 / 25 | 4022 |
| 20 / 50 | 2996 |
| 35 / 80 | 2268 |

![Edge methods](03-edge-detection/output/edge_methods_comparison.png)

## Testing

The tests include analytical small-array cases, independent reference loops, behavior checks, real-image output validation, and selected comparisons with `scikit-image`.

```bash
python scripts/validate_repository.py
```
