"""Five fixed-window filters, finite real 2D input, new unclipped float64 output.

NumPy reflect excludes the edge. Even windows anchor at floor(k/2), extra
sample above/left; even medians average the two central sorted samples.
"""
from fractions import Fraction
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _windows(image,window,mode,cval):
    a=np.asarray(image)
    if a.ndim!=2 or not a.size or a.dtype.kind not in 'buif':
        raise ValueError('Expected a nonempty real numeric 2D image')
    with np.errstate(over='ignore'):a=a.astype(np.float64)
    if not np.isfinite(a).all():raise ValueError('Image must be finite in float64')
    try:w=tuple(window)
    except TypeError as exc:raise ValueError('Two positive integer dimensions required') from exc
    if len(w)!=2 or any(isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) or k<1 for k in w):
        raise ValueError('Two positive integer dimensions required')
    if mode not in ('reflect','edge','constant'):raise ValueError('Unknown boundary mode')
    c=np.asarray(cval)
    if c.ndim or c.dtype.kind not in 'iuf' or not math.isfinite(float(cval)):
        raise ValueError('Finite real nonboolean cval required')
    pad=tuple((int(k)//2,int(k)-1-int(k)//2) for k in w)
    kwargs={'constant_values':float(cval)} if mode=='constant' else {}
    z=sliding_window_view(np.pad(a,pad,mode=mode,**kwargs),w).reshape(-1,math.prod(w))
    return a.shape,z


def _average(z):
    """Scaled averages, exact rational fallback for cancellation/subnormals.

    Exact binary fractions do not depend on extended longdouble availability.
    Internal negligible underflow is allowed; no epsilon changes the inputs.
    """
    scale=np.max(abs(z),axis=1)
    with np.errstate(under='ignore'):
        v=np.divide(z,scale[:,None],out=np.zeros_like(z),where=scale[:,None]!=0)
        avg=v.mean(axis=1);out=avg*scale
    fragile=((z.min(axis=1)<0)&(z.max(axis=1)>0)&(abs(avg)<=32*np.finfo(float).eps*np.mean(abs(v),axis=1)))
    fragile|=(scale>0)&(scale<np.finfo(float).tiny*z.shape[1])
    for i in np.flatnonzero(fragile):
        out[i]=float(sum((Fraction.from_float(float(x)) for x in z[i]),Fraction())/z.shape[1])
    constant=z.min(axis=1)==z.max(axis=1);out[constant]=z[constant,0]
    return out


def order_filter(image,method='median',window=(3,3),*,d=0,mode='reflect',cval=0.):
    """method: median/minimum/maximum/midpoint/alpha_trimmed.

    d is total symmetric trimming, an even integer 0<=d<mn. Nonzero d is
    rejected for other methods. Bool pixels are promoted, bool parameters fail.
    """
    if method not in ('median','minimum','maximum','midpoint','alpha_trimmed'):
        raise ValueError('Unknown method')
    shape,z=_windows(image,window,mode,cval);n=z.shape[1]
    if isinstance(d,(bool,np.bool_)) or not isinstance(d,(int,np.integer)) or d<0 or d>=n or d%2:
        raise ValueError('d must be an even integer with 0<=d<N')
    if method!='alpha_trimmed' and d:raise ValueError('d applies only to alpha_trimmed')
    if method=='minimum':out=z.min(axis=1)
    elif method=='maximum':out=z.max(axis=1)
    elif method=='midpoint':out=_average(np.column_stack((z.min(axis=1),z.max(axis=1))))
    else:
        z=np.sort(z,axis=1)
        if method=='median':out=z[:,n//2] if n%2 else _average(z[:,n//2-1:n//2+1])
        else:out=_average(z[:,d//2:n-d//2])
    if not np.isfinite(out).all():raise FloatingPointError('Nonfinite output')
    return out.reshape(shape).astype(np.float64,copy=True)


def median_filter(image,window=(3,3),*,mode='reflect',cval=0.):
    return order_filter(image,'median',window,mode=mode,cval=cval)


def minimum_filter(image,window=(3,3),*,mode='reflect',cval=0.):
    return order_filter(image,'minimum',window,mode=mode,cval=cval)


def maximum_filter(image,window=(3,3),*,mode='reflect',cval=0.):
    return order_filter(image,'maximum',window,mode=mode,cval=cval)


def midpoint_filter(image,window=(3,3),*,mode='reflect',cval=0.):
    return order_filter(image,'midpoint',window,mode=mode,cval=cval)


def alpha_trimmed_mean(image,window=(3,3),*,d=0,mode='reflect',cval=0.):
    return order_filter(image,'alpha_trimmed',window,d=d,mode=mode,cval=cval)
