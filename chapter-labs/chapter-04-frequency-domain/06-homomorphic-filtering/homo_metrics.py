"""Known-reflectance diagnostics with explicit global log-scale alignment."""
import numpy as np
from homomorphic import real_image
from homo_support import low_pass


def measure(output, truth):
    output, truth = real_image(output), real_image(truth)
    if output.shape != truth.shape or np.any(output <= 0) or np.any(truth <= 0):
        raise ValueError('Require same-shape positive output and truth.')
    a, b = np.log(output), np.log(truth)
    difference = a-b
    offset = float(difference.mean())
    residual = difference-offset
    lp = low_pass.gaussian_low_pass(output.shape, .03)
    low_residual = np.fft.ifft2(np.fft.fft2(residual)*lp).real
    hp = 1-low_pass.gaussian_low_pass(output.shape, .08)
    da = np.fft.ifft2(np.fft.fft2(a)*hp).real.std()
    db = np.fft.ifft2(np.fft.fft2(b)*hp).real.std()
    return dict(log_scale_offset=offset, log_reflectance_rmse=float(np.sqrt(np.mean(residual**2))),
                low_frequency_log_residual_std=float(low_residual.std()),
                detail_std_ratio=float(da/db) if db > 1e-14 else None,
                aligned_intensity_rmse=float(np.sqrt(np.mean((np.exp(a-offset)-truth)**2))),
                log_dynamic_range=float(np.ptp(a)), mean_intensity=float(output.mean()))
