"""Unclipped scientific measurements, separate from filtering and noise models."""
import math
import numpy as np


def quality(clean,observed,output,mask):
    clean,observed,output=(np.asarray(x,dtype=float) for x in [clean,observed,output])
    mask=np.asarray(mask)
    if clean.ndim!=2 or not clean.size or observed.shape!=clean.shape or output.shape!=clean.shape or mask.shape!=clean.shape or mask.dtype!=bool:
        raise ValueError('Equal nonempty 2D images and a boolean corruption mask required')
    if not all(np.isfinite(x).all() for x in [clean,observed,output]):
        raise ValueError('Finite metric inputs required')
    error=output-clean;before=observed-clean
    mse=float(np.mean(error*error));input_mse=float(np.mean(before*before))
    result=dict(mse=mse,rmse=math.sqrt(mse),mae=float(np.mean(abs(error))),
                psnr_db=None if mse==0 else 10*math.log10(1/mse),psnr_status='exact' if mse==0 else 'finite',
                residual_bias=float(np.mean(error)),input_mse=input_mse,
                residual_ratio=None if input_mse==0 else mse/input_mse,
                mse_corrupted=float(np.mean(error[mask]**2)) if mask.any() else None,
                mse_uncorrupted=float(np.mean(error[~mask]**2)) if (~mask).any() else None)
    if any(isinstance(v,(int,float)) and not math.isfinite(v) for v in result.values()):
        raise FloatingPointError('Metric is not representable')
    return result


def details(clean,output):
    dark=slice(6,32);bright=slice(46,74)
    clean_dark=float(np.mean((clean[35,dark]+clean[37,dark])/2-clean[36,dark]))
    clean_bright=float(np.mean(clean[44,bright]-(clean[43,bright]+clean[45,bright])/2))
    dark_contrast=float(np.mean((output[35,dark]+output[37,dark])/2-output[36,dark]))
    bright_contrast=float(np.mean(output[44,bright]-(output[43,bright]+output[45,bright])/2))
    threshold_dark=(clean[36,20]+(clean[35,20]+clean[37,20])/2)/2
    threshold_bright=(clean[44,60]+(clean[43,60]+clean[45,60])/2)/2
    return dict(dark_contrast_ratio=dark_contrast/clean_dark,bright_contrast_ratio=bright_contrast/clean_bright,
                dark_line_width=int(np.count_nonzero(output[31:42,20]<threshold_dark-1e-12)),
                bright_line_width=int(np.count_nonzero(output[39:50,60]>threshold_bright+1e-12)),
                bright_point_error=float(abs(output[28,15]-clean[28,15])),
                dark_point_error=float(abs(output[28,25]-clean[28,25])))


def edge_metrics(image):
    profile=image.mean(axis=0)
    def crossing(level):
        candidates=np.flatnonzero(profile>=level)
        if not candidates.size:return None
        k=int(candidates[0])
        if k==0:return 0.
        return k-1+float((level-profile[k-1])/(profile[k]-profile[k-1]))
    left,right,middle=crossing(.30),crossing(.70),crossing(.50)
    return dict(edge_width_10_90=right-left,edge_midpoint_shift=middle-39.5,
                edge_gradient_ratio=float(np.max(np.diff(profile))/.5))
