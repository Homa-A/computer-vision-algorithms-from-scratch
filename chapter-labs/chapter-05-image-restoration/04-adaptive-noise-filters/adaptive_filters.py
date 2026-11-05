"""Adaptive local shrinkage and exact two-stage adaptive median (original fallback).

All numerical outputs are new float64 arrays. No clipping or input mutation.
"""
from fractions import Fraction
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _scalar(value,name):
    v=np.asarray(value)
    if v.ndim or v.dtype.kind not in 'iuf' or not math.isfinite(float(value)):
        raise ValueError(name+' must be a finite real nonboolean scalar')
    return float(value)


def _image(image):
    a=np.asarray(image)
    if a.ndim!=2 or not a.size or a.dtype.kind not in 'buif':raise ValueError('Expected nonempty real numeric 2D image')
    with np.errstate(over='ignore'):a=a.astype(np.float64)
    if not np.isfinite(a).all():raise ValueError('Image must be finite in float64')
    return a


def _window(window):
    try:w=tuple(window)
    except TypeError as exc:raise ValueError('Two positive integer dimensions required') from exc
    if len(w)!=2 or any(isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) or k<1 for k in w):raise ValueError('Two positive integer dimensions required')
    return tuple(int(k) for k in w)


def _windows(a,window,mode,cval):
    w=_window(window);cval=_scalar(cval,'cval')
    if mode not in ('reflect','edge','constant'):raise ValueError('Unknown boundary mode')
    pad=tuple((k//2,k-1-k//2) for k in w)
    options={'constant_values':cval} if mode=='constant' else {}
    return sliding_window_view(np.pad(a,pad,mode=mode,**options),w).reshape(-1,math.prod(w))


def _statistics(z):
    """Shift by an actual sample, scaled centered population variance (ddof=0).

    Avoid E[z²]-E[z]² cancellation, overflow in sums and longdouble assumptions.
    A genuinely unrepresentable variance raises; variance underflow rounds to 0.
    """
    with np.errstate(over='ignore',under='ignore'):
        delta=z-z[:,0,None]
    if not np.isfinite(delta).all():raise FloatingPointError('Local variance not representable in float64')
    scale=np.max(abs(delta),axis=1)
    with np.errstate(under='ignore',over='ignore',invalid='ignore'):
        normalized=np.divide(delta,scale[:,None],out=np.zeros_like(delta),where=scale[:,None]!=0)
        offset=normalized.mean(axis=1)*scale
        mean=z[:,0]+offset;dev=delta-offset[:,None];s=np.max(abs(dev),axis=1)
        norm=np.divide(dev,s[:,None],out=np.zeros_like(dev),where=s[:,None]!=0)
        sd=s*np.sqrt(np.mean(norm*norm,axis=1))
        variance=sd*sd
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():raise FloatingPointError('Local statistics not representable in float64')
    return mean,variance


def local_statistics(image,window=(7,7),*,mode='reflect',cval=0.):
    """Population mean/variance, even anchor floor(k/2), NumPy reflect excludes edge."""
    a=_image(image);mean,var=_statistics(_windows(a,window,mode,cval))
    return mean.reshape(a.shape).copy(),var.reshape(a.shape).copy()


def adaptive_local_noise_filter(image,window=(7,7),*,noise_variance,mode='reflect',cval=0.,regularization='clipped',return_diagnostics=False):
    """literal: g−(noise_variance/local_variance)(g−mean); clipped: cap ratio at 1.

    Noise=0 gives exact identity. Positive noise and zero/underflowed variance
    give local mean. No epsilon. Unrepresentable result/diagnostic weight raises.
    """
    a=_image(image);nv=_scalar(noise_variance,'noise_variance')
    if nv<0 or np.asarray(noise_variance)<0:raise ValueError('noise_variance must be nonnegative')
    if regularization not in ('literal','clipped'):raise ValueError('Choose literal or clipped')
    z=_windows(a,window,mode,cval)
    if nv==0 and not return_diagnostics:return a.copy()
    mean,var=_statistics(z);g=a.ravel();weight=np.zeros_like(var)
    if nv==0:out=g.copy()
    else:
        zero=var==0;weight[zero]=1.
        with np.errstate(over='ignore',under='ignore'):weight[~zero]=nv/var[~zero]
        if regularization=='clipped':weight=np.minimum(weight,1.)
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):out=g-weight*(g-mean)
        out[zero]=mean[zero]
        # Exact rational fallback avoids spurious overflow from ratio*deviation.
        for i in np.flatnonzero(~np.isfinite(out)):
            if var[i]==0:continue
            correction=Fraction.from_float(nv)*(Fraction.from_float(float(g[i]))-Fraction.from_float(float(mean[i])))/Fraction.from_float(float(var[i]))
            try:out[i]=float(Fraction.from_float(float(g[i]))-correction)
            except OverflowError as exc:raise FloatingPointError('Literal output not representable') from exc
    if not np.isfinite(out).all():raise FloatingPointError('Output not representable')
    result=out.reshape(a.shape).copy()
    if not return_diagnostics:return result
    if not np.isfinite(weight).all():raise FloatingPointError('Diagnostic weight not representable')
    return result,dict(mean=mean.reshape(a.shape).copy(),variance=var.reshape(a.shape).copy(),weight=weight.reshape(a.shape).copy())


def estimate_noise_variance(image,region):
    """Observation-only ddof=0 variance in a predeclared rectangular flat ROI.

    region=(row_start,row_stop,col_start,col_stop), no clean image input.
    Assumes constant signal, stationary independent noise; edges/texture bias it.
    """
    a=_image(image)
    try:r=tuple(region)
    except TypeError as exc:raise ValueError('Four integer ROI bounds required') from exc
    if len(r)!=4 or any(isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) for k in r):raise ValueError('Four integer ROI bounds required')
    y0,y1,x0,x1=r
    if not (0<=y0<y1<=a.shape[0] and 0<=x0<x1<=a.shape[1]):raise ValueError('ROI outside image')
    _,v=_statistics(a[y0:y1,x0:x1].reshape(1,-1));return float(v[0])


def adaptive_median_filter(image,initial_size=3,max_size=7,*,mode='reflect',cval=0.,return_diagnostics=False):
    """Strict Stage A/B, odd square growth by two, fallback ORIGINAL center.

    All windows use the original observation. Diagnostic sizes count the last
    attempted window; A failures=expansions+fallback, not just final fallbacks.
    """
    a=_image(image)
    for s in (initial_size,max_size):
        if isinstance(s,(bool,np.bool_)) or not isinstance(s,(int,np.integer)) or s<1 or s%2==0:raise ValueError('Sizes must be positive odd integers')
    # NumPy narrow scalars can wrap at max_size+1 before Python range sees it.
    initial_size,max_size=int(initial_size),int(max_size)
    if initial_size>max_size:raise ValueError('initial_size exceeds max_size')
    g=a.ravel();out=g.copy();pending=np.ones(len(g),bool)
    size=np.full(len(g),initial_size,np.int64);passed=np.zeros(len(g),bool);replaced=passed.copy()
    lo=np.empty_like(g);hi=np.empty_like(g);med=np.empty_like(g)
    for k in range(initial_size,max_size+1,2):
        z=_windows(a,(k,k),mode,cval);minimum=z.min(axis=1);maximum=z.max(axis=1);middle=np.partition(z,z.shape[1]//2,axis=1)[:,z.shape[1]//2]
        size[pending]=k;lo[pending]=minimum[pending];hi[pending]=maximum[pending];med[pending]=middle[pending]
        # Comparisons are equivalent to strict signed differences without overflow.
        accept=pending&(middle>minimum)&(middle<maximum)
        change=accept&~((g>minimum)&(g<maximum))
        out[change]=middle[change];passed[accept]=True;replaced[change]=True;pending[accept]=False
        if not pending.any():break
    result=out.reshape(a.shape).copy()
    if not return_diagnostics:return result
    reshape=lambda v:v.reshape(a.shape).copy()
    expansions=(size-initial_size)//2
    return result,dict(size=reshape(size),expansions=reshape(expansions),stage_a_passed=reshape(passed),stage_b_replaced=reshape(replaced),
        retained=reshape(~replaced),fallback=reshape(pending),stage_a_failures=reshape(expansions+pending.astype(np.int64)),
        minimum=reshape(lo),median=reshape(med),maximum=reshape(hi))
