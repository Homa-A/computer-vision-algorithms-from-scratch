"""Slow independent boundary and scalar oracles, including Decimal stress checks."""
from decimal import Decimal, localcontext, MAX_EMAX, MIN_EMIN
from fractions import Fraction
import math
import numpy as np


def coordinate(index,length,mode):
    if 0<=index<length:return index
    if mode=='constant':return None
    if mode=='edge':return min(max(index,0),length-1)
    if mode!='reflect':raise ValueError('Unknown boundary mode')
    if length==1:return 0
    period=2*(length-1);index=index%period
    return index if index<length else period-index


def scalar_mean(values,method,Q=1.5):
    """Ordinary-scale independent formulas. Stress cases use decimal_mean."""
    values=[float(v) for v in values];n=len(values)
    if method=='arithmetic' or (method=='contra' and Q==0):return math.fsum(values)/n
    if min(values)<0:raise ValueError('Nonnegative domain')
    if method=='geometric':
        if 0 in values:return 0.
        # Direct product is intentionally independent at moderate reference scales.
        return math.prod(values)**(1/n)
    if method=='harmonic' or (method=='contra' and Q==-1):
        if 0 in values:return 0.
        return n/math.fsum(1/v for v in values)
    if method!='contra':raise ValueError('Unknown method')
    if all(v==0 for v in values) or (Q<0 and 0 in values):return 0.
    return math.fsum(v**(Q+1) for v in values)/math.fsum(v**Q for v in values)


def filter_reference(image,method,window=(3,3),*,mode='reflect',cval=0.,Q=1.5):
    image=np.asarray(image,dtype=float);height,width=image.shape;kh,kw=window
    output=np.empty_like(image)
    for y,x in np.ndindex(image.shape):
        values=[]
        for i,j in np.ndindex(window):
            yy=coordinate(y+i-kh//2,height,mode);xx=coordinate(x+j-kw//2,width,mode)
            values.append(cval if yy is None or xx is None else image[yy,xx])
        output[y,x]=scalar_mean(values,method,Q)
    return output


def decimal_mean(values,method,Q=1.5):
    """100-digit scalar oracle; no NumPy log/power/filter implementation reused.

    Decimal.from_float treats the actual binary inputs exactly. Normalized
    Decimal powers support the selected large-range/Q cases without longdouble.
    """
    with localcontext() as ctx:
        ctx.prec=100;ctx.Emax=MAX_EMAX;ctx.Emin=MIN_EMIN
        z=[Decimal.from_float(float(v)) for v in values];n=Decimal(len(z))
        if method=='arithmetic' or (method=='contra' and Q==0):return float(sum(z)/n)
        if any(v<0 for v in z):raise ValueError('Nonnegative domain')
        if method=='geometric':
            if any(v==0 for v in z):return 0.
            return float((sum(v.ln() for v in z)/n).exp())
        if method=='harmonic' or (method=='contra' and Q==-1):
            if any(v==0 for v in z):return 0.
            return float(n/sum(1/v for v in z))
        q=Decimal.from_float(float(Q))
        if (q<0 and any(v==0 for v in z)) or max(z)==0:return 0.
        # Selected finite stress Q values fit Decimal exponent bounds. Huge-Q
        # limit tests use analytic concentration bounds instead of huge powers.
        return float(sum(v**(q+1) for v in z)/sum(v**q for v in z))
