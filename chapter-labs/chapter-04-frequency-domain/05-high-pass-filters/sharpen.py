"""High-pass and sharpening: cycles/pixel, backward normalization, complex128."""
from numbers import Real
import numpy as np
from shared import lp

def nonnegative(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real):
        raise ValueError("Strength must be a nonnegative real scalar.")
    try: value=float(value)
    except (OverflowError,ValueError) as exc: raise ValueError("Outside float64 range.") from exc
    if not np.isfinite(value) or value<0: raise ValueError("Strength must be finite and nonnegative.")
    return value

def high_pass(shape,cutoff,kind="gaussian",*,order=2,centered=False):
    """Ideal rejects D<=cutoff; every complement has exact zero DC."""
    return 1-lp.transfer_function(shape,cutoff,kind,order=order,centered=centered)

def laplacian_response(shape,*,centered=False,continuous=False):
    if not isinstance(continuous,(bool,np.bool_)): raise ValueError("continuous must be boolean.")
    fy,fx=lp.frequency_axes(shape,centered=centered)
    return -4*np.pi**2*(fy[:,None]**2+fx[None,:]**2) if continuous else -4*(np.sin(np.pi*fy[:,None])**2+np.sin(np.pi*fx[None,:])**2)

def spatial_laplacian(image):
    """Independent scalar stencil with periodic boundary indexing, no FFT."""
    x=lp._image(image);m,n=x.shape;out=np.empty_like(x)
    for i,j in np.ndindex(x.shape):
        out[i,j]=x[(i+1)%m,j]+x[(i-1)%m,j]+x[i,(j+1)%n]+x[i,(j-1)%n]-4*x[i,j]
    return out

def response(shape,cutoff=.08,kind="gaussian",*,method="hp",order=2,k=1,A=2,a=1,b=1,s=.2,centered=False):
    k,A,a,b,s=map(nonnegative,(k,A,a,b,s))
    if A<1: raise ValueError("A must be >=1.")
    h=high_pass(shape,cutoff,kind,order=order,centered=centered)
    if method=="hp": result=h
    elif method=="unsharp": result=1+k*h
    elif method=="boost": result=(A-1)+h
    elif method=="emphasis": result=a+b*h
    elif method=="laplacian": result=1-s*laplacian_response(shape,centered=centered)
    else: raise ValueError("Unknown method.")
    if not np.isfinite(result).all(): raise ValueError("Response overflow.")
    return result

def apply(image,cutoff=.08,kind="gaussian",*,padding=(0,0),boundary="zero",centered=False,**parameters):
    """Pad py/px at BOTH ends; crop starts (py,px); measure full inverse residue."""
    real=not np.iscomplexobj(np.asarray(image));x=lp._image(image)
    py,px=lp._pair(padding,"padding",minimum=0)
    if boundary not in ("zero","reflect"): raise ValueError("Unknown boundary.")
    padded=np.pad(x,((py,py),(px,px)),mode="constant" if boundary=="zero" else "reflect")
    h=response(padded.shape,cutoff,kind,centered=centered,**parameters)
    full=lp.apply_transfer(padded,h,centered=centered)
    if not np.isfinite(full).all(): raise ValueError("Filtered output overflow.")
    residue=float(abs(full.imag).max()) if real else None
    return lp.FilterResult(full[py:py+x.shape[0],px:px+x.shape[1]].copy(),residue,padded.shape,(py,px))
