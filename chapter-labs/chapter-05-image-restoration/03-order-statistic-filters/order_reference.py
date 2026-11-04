"""Independent literal neighborhoods and exact binary-rational arithmetic.

No production import, np.pad, sliding windows or filtering library calls.
"""
from fractions import Fraction
import numpy as np


def index(i,n,mode):
    if 0<=i<n:return i
    if mode=='constant':return None
    if mode=='edge':return min(n-1,max(0,i))
    if n==1:return 0
    i%=2*(n-1)
    return i if i<n else 2*(n-1)-i


def samples(a,y,x,window,mode='reflect',cval=0.):
    z=[]
    for dy in range(window[0]):
        for dx in range(window[1]):
            yy=index(y+dy-window[0]//2,len(a),mode)
            xx=index(x+dx-window[1]//2,a.shape[1],mode)
            z.append(float(cval if yy is None or xx is None else a[yy,xx]))
    return sorted(z)


def value(z,method,d=0):
    z=sorted(z);n=len(z)
    if method=='minimum':return z[0]
    if method=='maximum':return z[-1]
    if method=='median':z=[z[n//2]] if n%2 else z[n//2-1:n//2+1]
    elif method=='midpoint':z=[z[0],z[-1]]
    elif method=='alpha_trimmed':z=z[d//2:n-d//2]
    elif method!='arithmetic':raise ValueError('Unknown reference method')
    return float(sum((Fraction.from_float(float(x)) for x in z),Fraction())/len(z))


def reference(image,method,window=(3,3),*,d=0,mode='reflect',cval=0.,points=None):
    a=np.asarray(image,dtype=float)
    if points is not None:return np.array([value(samples(a,y,x,window,mode,cval),method,d) for y,x in points])
    out=np.empty_like(a)
    for y,x in np.ndindex(a.shape):out[y,x]=value(samples(a,y,x,window,mode,cval),method,d)
    return out
