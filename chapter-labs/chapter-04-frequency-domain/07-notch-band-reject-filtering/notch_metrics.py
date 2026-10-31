"""Known-truth diagnostics; separate noise suppression from signal damage."""
import numpy as np
from notch_support import low_pass


def _gradient(x):
    return np.roll(x,-1,axis=0)-x, np.roll(x,-1,axis=1)-x


def noise_region(shape, components, width=.02):
    fy,fx = low_pass.frequency_axes(shape)
    region = np.zeros(shape,dtype=bool)
    for component in components:
        for sign in [1,-1]:
            y,x = np.array(component['frequency_cpp'])*sign
            dy=np.abs(fy-y);dy=np.minimum(dy,1-dy)
            dx=np.abs(fx-x);dx=np.minimum(dx,1-dx)
            region |= np.hypot(dy[:,None],dx[None,:])<=width
    return region


def evaluate(clean,noisy,periodic,components,response):
    """Unit-range PSNR, true periodic-component gain and clean-transfer damage.

    clean and periodic are independently filtered with H to exploit linearity.
    Off-noise spectral error excludes a .02 cpp union around true noise and the
    .025 cpp DC neighborhood. No PSNR-only perceptual conclusion is intended.
    """
    fc = np.fft.fft2(clean); fn=np.fft.fft2(periodic)
    full=np.fft.ifft2(np.fft.fft2(noisy)*response)
    output=full.real
    clean_output=np.fft.ifft2(fc*response).real
    periodic_output=np.fft.ifft2(fn*response).real
    error=output-clean;mse=float(np.mean(error**2))
    region=noise_region(clean.shape,components)
    denom=float(np.linalg.norm(fn[region]))
    gain=float(np.linalg.norm((fn*response)[region])/denom) if denom else 0.
    outside=~region & (low_pass.radial_distance(clean.shape)>.025)
    signal_norm=float(np.linalg.norm(fc[outside]))
    off_error=float(np.linalg.norm((fc*response-fc)[outside])/signal_norm) if signal_norm else 0.
    cy,cx=_gradient(clean);oy,ox=_gradient(output);fy,fx=_gradient(clean_output)
    grad_norm=float(np.sqrt(np.mean(cy**2+cx**2)))
    return dict(mse=mse,psnr_db=float(-10*np.log10(mse)) if mse>0 else None,
                psnr_infinite=bool(mse==0),noise_spectral_gain=gain,
                attenuation_db=float(20*np.log10(gain)) if gain>0 else None,
                periodic_rms_gain=float(np.linalg.norm(periodic_output)/np.linalg.norm(periodic)) if np.any(periodic) else 0.,
                off_noise_clean_spectral_error=off_error,
                clean_gradient_ratio=float(np.sqrt(np.mean(fy**2+fx**2))/grad_norm) if grad_norm else None,
                gradient_error_rmse=float(np.sqrt(np.mean((oy-cy)**2+(ox-cx)**2))),
                imaginary_residual=float(np.max(np.abs(full.imag))))


def ringing(response):
    m,n=response.shape
    step=np.broadcast_to((np.arange(n)>=n//2).astype(float),(m,n))
    out=np.fft.ifft2(np.fft.fft2(step)*response).real
    kernel=np.fft.ifft2(response).real
    y=np.minimum(np.arange(m),m-np.arange(m));x=np.minimum(np.arange(n),n-np.arange(n))
    far=np.hypot(y[:,None],x[None,:])>4
    return dict(step_overshoot=float(max(0.,out.max()-1)),step_undershoot=float(max(0.,-out.min())),
                impulse_tail_energy_ratio=float(np.sum(kernel[far]**2)/np.sum(kernel**2)))
