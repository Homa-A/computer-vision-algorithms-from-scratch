"""Valid real-image correlation, stable ZNCC and deterministic localization.

Top-left (row,column), unscaled forward FFT / inverse 1/(PQ). Invalid ZNCC
windows carry finite zero payloads and false masks; selection ALWAYS uses masks.
"""
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral,Real
import numpy as np

EPS=np.finfo(np.float64).eps


def image_array(value):
    raw=np.asarray(value)
    if raw.ndim!=2 or raw.size==0:raise ValueError('Expected a nonempty 2D array.')
    if raw.dtype.kind not in 'buif':raise TypeError('Expected real numeric grayscale data.')
    with np.errstate(over='ignore',invalid='ignore'):a=raw.astype(np.float64)
    if not np.isfinite(a).all():raise ValueError('Values must be finite in float64.')
    return a


def inputs(image,template):
    a,t=image_array(image),image_array(template)
    if any(v>s for v,s in zip(t.shape,a.shape)):raise ValueError('Template is larger than image.')
    return a,t


def pair(value,minimum=0):
    try:values=tuple(value)
    except TypeError as exc:raise ValueError('Expected two integer dimensions.') from exc
    if len(values)!=2 or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Integral) or v<minimum for v in values):
        raise ValueError('Expected two integer dimensions within the permitted range.')
    return tuple(map(int,values))


@dataclass(frozen=True)
class Response:
    values: np.ndarray
    valid: np.ndarray
    imaginary_residual: float | None
    fft_shape: tuple[int,int] | None
    full: np.ndarray | None = None
    fallback_windows: int = 0


def direct_correlation(image,template):
    """Independent spatial valid dot products; no FFT or template reversal."""
    a,t=inputs(image,template);h,w=t.shape
    out=np.empty((a.shape[0]-h+1,a.shape[1]-w+1))
    with np.errstate(over='raise',invalid='raise'):
        for y,x in np.ndindex(out.shape):out[y,x]=np.sum(a[y:y+h,x:x+w]*t,dtype=np.float64)
    if not np.isfinite(out).all():raise FloatingPointError('Raw correlation overflow.')
    return out


def direct_ssd(image,template):
    """Independent window SSD (lower is better); matching uses its negative."""
    a,t=inputs(image,template);h,w=t.shape
    out=np.empty((a.shape[0]-h+1,a.shape[1]-w+1))
    with np.errstate(over='raise',invalid='raise'):
        for y,x in np.ndindex(out.shape):out[y,x]=np.sum((a[y:y+h,x:x+w]-t)**2)
    if not np.isfinite(out).all():raise FloatingPointError('SSD overflow.')
    return out


def fft_correlation(image,template,*,fft_shape=None):
    """Linear convolution with reversed T; full[h-1:H,w-1:W] is valid C.

    The public full-map contract requires P>=H+h-1,Q>=W+w-1. Oversized FFTs
    are allowed; full is cropped to the minimal full shape, not shifted.
    """
    a,t=inputs(image,template);H,W=a.shape;h,w=t.shape
    minimum=(H+h-1,W+w-1);shape=minimum if fft_shape is None else pair(fft_shape,1)
    if any(s<m for s,m in zip(shape,minimum)):raise ValueError('Insufficient padding for a full linear map.')
    sa,ea=_power_scale(a);st,et=_power_scale(t)
    with np.errstate(over='raise',invalid='raise'):
        inverse=np.fft.ifft2(np.fft.fft2(sa,s=shape)*np.fft.fft2(st[::-1,::-1],s=shape))
    with np.errstate(over='raise',invalid='raise',under='ignore'):
        inverse=np.ldexp(inverse.real,ea+et)+1j*np.ldexp(inverse.imag,ea+et)
    if not np.isfinite(inverse).all():raise FloatingPointError('FFT correlation overflow.')
    residual=float(np.max(abs(inverse.imag)));full=inverse.real[:minimum[0],:minimum[1]].copy()
    values=full[h-1:H,w-1:W].copy()
    return Response(values,np.ones(values.shape,bool),residual,shape,full)


def circular_correlation(image,template):
    """All image-sized circular lags; valid top-left lags do not wrap.

    This demonstration is deliberately separate from the strict linear API.
    Output[y,x]=sum I[(y+i)%H,(x+j)%W]*T[i,j]. No fftshift is used.
    """
    a,t=inputs(image,template)
    sa,ea=_power_scale(a);st,et=_power_scale(t)
    with np.errstate(over='raise',invalid='raise'):
        out=np.fft.ifft2(np.fft.fft2(sa)*np.conj(np.fft.fft2(st,s=a.shape)))
    with np.errstate(over='raise',invalid='raise',under='ignore'):
        out=np.ldexp(out.real,ea+et)+1j*np.ldexp(out.imag,ea+et)
    if not np.isfinite(out).all():raise FloatingPointError('Circular correlation overflow.')
    return out


def _power_scale(a):
    largest=float(np.max(np.abs(a)))
    exponent=int(np.frexp(largest)[1]) if largest else 0
    with np.errstate(under='ignore'):scaled=np.ldexp(a,-exponent)
    return scaled,exponent


def _unit_center(a):
    """Robust patch standardization, including near-constant huge/tiny values."""
    if np.all(a==a.flat[0]):return None
    scaled,_=_power_scale(a)
    difference=scaled-scaled.flat[0]
    magnitude=float(np.max(abs(difference)))
    if not magnitude:raise FloatingPointError('Nonconstant patch lost representable contrast.')
    difference/=magnitude
    centered=difference-difference.mean()
    norm=float(np.linalg.norm(centered))
    if norm==0 or not np.isfinite(norm):raise FloatingPointError('Undefined centered norm.')
    return centered/norm


def _axis_moments(mean,m2,count,length,axis):
    """Chan parallel-moment merges over sliding dyadic blocks, no Q-S²/K."""
    @lru_cache(None)
    def stats(n):
        if n==1:return mean,m2,count
        left=n//2;right=n-left
        am,av,an=stats(left);bm,bv,bn=stats(right)
        size=mean.shape[axis]-n+1
        first=[slice(None)]*2;second=list(first)
        first[axis]=slice(0,size);second[axis]=slice(left,left+size)
        am,av=am[tuple(first)],av[tuple(first)];bm,bv=bm[tuple(second)],bv[tuple(second)]
        delta=bm-am;total=an+bn
        with np.errstate(under='ignore'):
            return am+delta*(bn/total),av+bv+delta*delta*(an*bn/total),total
    result=stats(length)[:2]
    stats.cache_clear()
    # Break the recursive closure cycle so large temporary arrays are released
    # now, rather than waiting for a later cyclic-garbage-collection pass.
    stats=None
    return result


def _window_moments(a,shape):
    """O(HW(log h+log w)) positive centered moments on bounded scaled input."""
    h,w=shape
    mean,m2=_axis_moments(a,np.zeros_like(a),1,w,1)
    return _axis_moments(mean,m2,w,h,0)


def zncc(image,template):
    """FFT numerator and stable sliding centered moments, with precision fallback.

    No subtraction of raw sums of squares. Low-contrast windows whose FFT
    numerator cannot support ~1e-10 score accuracy are recomputed locally.
    Exact constant T raises; constant image windows have false validity masks.
    Unit scores are clipped only within 1e-10 roundoff; larger violations raise.
    """
    a,t=inputs(image,template);q=_unit_center(t)
    if q is None:raise ValueError('ZNCC is undefined for a zero-variance template.')
    scaled,_=_power_scale(a);scaled=scaled-scaled.flat[0]
    mean,m2=_window_moments(scaled,t.shape)
    numerator=fft_correlation(scaled,q)
    centered_dot=numerator.values-mean*float(np.sum(q))
    norm=np.sqrt(m2)
    # A conservative absolute FFT-dot roundoff scale, not a signal threshold.
    bound=128*EPS*max(1.,np.log2(np.prod(numerator.fft_shape)))*np.sqrt(t.size)*float(np.max(abs(scaled)))
    uncertain=(norm<=bound/1e-10)|(norm==0)
    values=np.zeros(norm.shape);valid=np.ones(norm.shape,bool)
    safe=~uncertain
    values[safe]=centered_dot[safe]/norm[safe]
    h,w=t.shape
    for y,x in zip(*np.where(uncertain)):
        patch=_unit_center(a[y:y+h,x:x+w])
        if patch is None:valid[y,x]=False
        else:values[y,x]=float(np.sum(patch*q))
    if not np.isfinite(values).all():raise FloatingPointError('Nonfinite ZNCC.')
    if np.any(abs(values[valid])>1+1e-10):raise FloatingPointError('ZNCC exceeded its numerical range.')
    values[valid]=np.clip(values[valid],-1,1)
    return Response(values,valid,numerator.imaginary_residual,numerator.fft_shape,
                    fallback_windows=int(np.count_nonzero(uncertain)))


def _tolerance(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real):raise ValueError(name+' must be finite nonnegative.')
    try:value=float(value)
    except (ValueError,OverflowError) as exc:raise ValueError(name+' is outside float64.') from exc
    if not np.isfinite(value) or value<0:raise ValueError(name+' must be finite nonnegative.')
    return value


def select_peaks(response,*,top_k=2,exclusion=(0,0),tie_atol=1e-12,tie_rtol=4096*EPS):
    """Row-major among numerical ties; rectangular NMS in response pixels."""
    if isinstance(top_k,(bool,np.bool_)) or not isinstance(top_k,Integral) or top_k<1:raise ValueError('top_k must be a positive integer.')
    ry,rx=pair(exclusion);atol=_tolerance(tie_atol,'tie_atol');rtol=_tolerance(tie_rtol,'tie_rtol')
    values=image_array(response.values);mask=np.asarray(response.valid)
    if mask.dtype!=bool or mask.shape!=values.shape:raise ValueError('Validity mask must be boolean and shape compatible.')
    eligible=mask.copy();peaks=[]
    for _ in range(top_k):
        if not eligible.any():break
        maximum=float(np.max(values[eligible]));tol=atol+rtol*abs(maximum)
        if not np.isfinite(tol):raise ValueError('Combined tie tolerance overflow.')
        choices=eligible & (values>=maximum-tol)
        y,x=np.argwhere(choices)[0];y,x=int(y),int(x)
        peaks.append(dict(row=y,column=x,score=float(values[y,x])))
        eligible[max(0,y-ry):y+ry+1,max(0,x-rx):x+rx+1]=False
    return peaks


def localize(response,*,top_k=2,exclusion=(0,0),tie_atol=1e-12,tie_rtol=4096*EPS):
    peaks=select_peaks(response,top_k=top_k,exclusion=exclusion,tie_atol=tie_atol,tie_rtol=tie_rtol)
    count=int(np.count_nonzero(response.valid))
    if not peaks:return dict(location=None,score=None,eligible_count=count,peaks=[],tie_count=0,gap=None,psr=None,psr_status='no_valid',ambiguous=False)
    best=peaks[0];y,x=best['row'],best['column'];values=response.values
    tol=float(tie_atol)+float(tie_rtol)*abs(best['score'])
    with np.errstate(over='ignore'):
        ties=int(np.count_nonzero(response.valid & (abs(values-best['score'])<=tol)))
    side=response.valid.copy();ry,rx=pair(exclusion)
    side[max(0,y-ry):y+ry+1,max(0,x-rx):x+rx+1]=False
    sidelobes=values[side];psr=None
    if not sidelobes.size:status='empty'
    else:
        scale=max(1.,abs(best['score']),float(np.max(abs(sidelobes))))
        side_scaled=sidelobes/scale;std=float(np.std(side_scaled))
        status='constant_or_precision_floor' if std<=np.sqrt(EPS) else 'defined'
        if status=='defined':psr=float((best['score']/scale-np.mean(side_scaled))/std)
    gap=None if len(peaks)<2 else float(max(0,best['score']-peaks[1]['score']))
    if gap is not None and not np.isfinite(gap):gap=None
    if gap is not None and gap<=tol:gap=0.
    return dict(location=[y,x],score=best['score'],eligible_count=count,peaks=peaks,
                tie_count=ties,gap=gap,psr=psr,psr_status=status,ambiguous=ties>1 or gap==0.)


@dataclass(frozen=True)
class Match:
    location: tuple[int,int] | None
    score: float | None
    response: np.ndarray
    validity: np.ndarray
    eligible_count: int
    diagnostics: dict
    numerical: Response


def match_template(image,template,*,method='zncc',top_k=2,exclusion=None,
                   tie_atol=1e-12,tie_rtol=4096*EPS):
    a,t=inputs(image,template)
    if method=='zncc':response=zncc(a,t)
    elif method=='fft':response=fft_correlation(a,t)
    elif method in ('direct','ssd'):
        values=direct_correlation(a,t) if method=='direct' else -direct_ssd(a,t)
        response=Response(values,np.ones(values.shape,bool),None,None)
    else:raise ValueError('Unknown matching method.')
    if exclusion is None:exclusion=(t.shape[0]//2,t.shape[1]//2)
    info=localize(response,top_k=top_k,exclusion=exclusion,tie_atol=tie_atol,tie_rtol=tie_rtol)
    return Match(tuple(info['location']) if info['location'] is not None else None,info['score'],
                 response.values,response.valid,info['eligible_count'],info,response)
