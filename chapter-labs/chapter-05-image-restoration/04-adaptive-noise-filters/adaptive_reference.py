"""Independent coordinate-enumerating exact rational local/stage reference."""
from fractions import Fraction as F
import numpy as np


def coordinate(i,n,mode):
    if 0<=i<n:return i
    if mode=='constant':return None
    if mode=='edge':return max(0,min(n-1,i))
    if n==1:return 0
    i%=2*(n-1);return i if i<n else 2*(n-1)-i


def neighborhood(a,y,x,w,mode,cval):
    z=[]
    for dy in range(w[0]):
        for dx in range(w[1]):
            yy=coordinate(y+dy-w[0]//2,len(a),mode);xx=coordinate(x+dx-w[1]//2,a.shape[1],mode)
            z.append(float(cval if yy is None or xx is None else a[yy,xx]))
    return z


def local_pixel(a,y,x,w,nv,policy,mode='reflect',cval=0.):
    values=[F.from_float(v) for v in neighborhood(a,y,x,w,mode,cval)]
    m=sum(values,F())/len(values);v=sum(((z-m)**2 for z in values),F())/len(values);g=F.from_float(float(a[y,x]));noise=F.from_float(float(nv))
    weight=F() if noise==0 else F(1) if v==0 else noise/v
    if policy=='clipped':weight=min(F(1),weight)
    out=g if noise==0 else m if v==0 else g-weight*(g-m)
    return float(out),dict(mean=float(m),variance=float(v),weight=float(weight))


def median_pixel(a,y,x,initial,maximum,mode='reflect',cval=0.):
    """Recursive per-pixel oracle, distinct from production's vectorized masks."""
    g=float(a[y,x])
    def visit(k):
        z=sorted(neighborhood(a,y,x,(k,k),mode,cval));lo,hi,med=z[0],z[-1],z[len(z)//2]
        success=lo<med<hi
        if not success and k<maximum:return visit(k+2)
        replacement=success and not lo<g<hi
        expansions=(k-initial)//2
        return med if replacement else g,dict(size=k,expansions=expansions,stage_a_passed=success,stage_b_replaced=replacement,
            retained=not replacement,fallback=not success,stage_a_failures=expansions+int(not success),minimum=lo,median=med,maximum=hi)
    return visit(initial)


def reference(a,method,params,points=None):
    a=np.asarray(a,dtype=float);points=list(np.ndindex(a.shape)) if points is None else points
    results=[];diagnostics=[]
    for y,x in points:
        if method=='local':out,d=local_pixel(a,y,x,params['window'],params['noise_variance'],params['regularization'],params['mode'],params['cval'])
        else:out,d=median_pixel(a,y,x,params['initial_size'],params['max_size'],params['mode'],params['cval'])
        results.append(out);diagnostics.append(d)
    return np.array(results),diagnostics
