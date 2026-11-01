"""Independent direct local-window mathematical references; no FFT matcher calls."""
import numpy as np


def unit(a):
    # Subtract the minimum before division; Sterbenz-safe for near constants.
    a=np.asarray(a,dtype=float)
    if np.all(a==a.flat[0]):return None
    if a.min()<0<a.max() and a.max()>np.finfo(float).max+a.min():a=a*.5
    with np.errstate(over='raise',under='ignore'):
        shifted=a-float(a.min())
    shifted=shifted/float(shifted.max());shifted-=np.sum(shifted)/shifted.size
    return shifted/np.sqrt(np.sum(shifted*shifted))


def maps(image,template):
    h,w=template.shape;shape=(image.shape[0]-h+1,image.shape[1]-w+1)
    q=unit(template)
    if q is None:raise ValueError('Undefined ZNCC template')
    windows=np.lib.stride_tricks.sliding_window_view(image,template.shape)
    raw=np.einsum('yxij,ij->yx',windows,template)
    ssd=-np.sum((windows-template)**2,axis=(-2,-1))
    shifted=windows-windows.min(axis=(-2,-1),keepdims=True)
    spread=shifted.max(axis=(-2,-1),keepdims=True)
    valid=spread[...,0,0]>0
    normalized=shifted/np.where(spread>0,spread,1.)
    normalized-=normalized.mean(axis=(-2,-1),keepdims=True)
    norms=np.sqrt(np.sum(normalized**2,axis=(-2,-1),keepdims=True))
    normalized/=np.where(norms>0,norms,1.)
    z=np.einsum('yxij,ij->yx',normalized,q)
    z[~valid]=0.
    return dict(direct=raw,fft=raw,ssd=ssd,zncc=z),valid



def full_convolution(a,t):
    h,w=t.shape;full=np.zeros((a.shape[0]+h-1,a.shape[1]+w-1))
    for y,x in np.ndindex(a.shape):
        for i,j in np.ndindex(t.shape):full[y+i,x+j]+=a[y,x]*t[h-1-i,w-1-j]
    return full


def circular(a,t):
    h,w=t.shape;out=np.zeros(a.shape)
    for y,x in np.ndindex(a.shape):
        for i,j in np.ndindex(t.shape):out[y,x]+=a[(y+i)%a.shape[0],(x+j)%a.shape[1]]*t[i,j]
    return out


def localization(values,valid,exclusion,top_k=2):
    available=valid.copy();peaks=[];ry,rx=exclusion
    for _ in range(top_k):
        if not available.any():break
        maximum=max(values[available]);tolerance=1e-12+4096*np.finfo(float).eps*abs(maximum)
        chosen=next((p for p in np.ndindex(values.shape) if available[p] and values[p]>=maximum-tolerance))
        y,x=chosen;peaks.append(dict(row=int(y),column=int(x),score=float(values[chosen])))
        yy,xx=np.indices(values.shape);available[(abs(yy-y)<=ry)&(abs(xx-x)<=rx)]=False
    if not peaks:return dict(location=None,score=None,eligible_count=int(valid.sum()),peaks=[],tie_count=0,gap=None,psr=None,psr_status='no_valid',ambiguous=False)
    y,x=peaks[0]['row'],peaks[0]['column'];score=float(values[y,x]);tol=1e-12+4096*np.finfo(float).eps*abs(score)
    ties=int(np.sum(valid&(np.abs(values-score)<=tol)));yy,xx=np.indices(values.shape)
    side=values[valid & ((abs(yy-y)>ry)|(abs(xx-x)>rx))];psr=None
    if not side.size:status='empty'
    else:
        scale=max(1.,abs(score),float(np.max(abs(side))));centered=side/scale;std=np.sqrt(np.mean((centered-centered.mean())**2))
        status='constant_or_precision_floor' if std<=np.sqrt(np.finfo(float).eps) else 'defined'
        if status=='defined':psr=float((score/scale-centered.mean())/std)
    gap=None if len(peaks)<2 else max(0.,score-peaks[1]['score'])
    if gap is not None and gap<=tol:gap=0.
    return dict(location=[y,x],score=score,eligible_count=int(valid.sum()),peaks=peaks,tie_count=ties,gap=gap,psr=psr,psr_status=status,ambiguous=ties>1 or gap==0.)


def scene_truth():
    def template(shape=(9,11)):
        y,x=np.indices(shape);h,w=shape
        a=np.full(shape,.12)
        a+=.55*((x>=w//5)&(x<=w//2)&(y>=1)&(y<h-1))
        a+=.3*((y>=h//2)&(y<=h//2+1)&(x>=w//3)&(x<w-1))
        a+=.75*((x==w-2)&(y==1))
        return a
    t=template();result={}
    names=['unique','top_left','bottom_right','odd_even','repeated','distractors','affine','brightness_decoy','noise','occlusion','rotated','scaled','near_constant','low_texture','absent']
    for name in names:
        shape=(47,61) if name=='odd_even' else (48,64)
        q=template((8,10)) if name=='odd_even' else t.copy()
        if name=='near_constant':q=2+1e-10*(q-q.mean())
        locations={'top_left':[[0,0]],'bottom_right':[[39,53]],'odd_even':[[29,38]],'repeated':[[4,5],[29,42]],'distractors':[[27,35]],'brightness_decoy':[[5,7]],'rotated':[[20,30]],'scaled':[[20,28]],'absent':[]}.get(name,[[18,27]])
        seed=899 if name=='absent' else 808
        a=.08+np.random.default_rng(seed).normal(0,.02,shape)
        for y,x in locations:
            obj=q.copy()
            if name in ['affine','brightness_decoy']:obj=1.8*obj+.4
            if name=='rotated':obj=np.rot90(obj)
            if name=='scaled':obj=obj.repeat(2,axis=0).repeat(2,axis=1)
            if name=='occlusion':obj[:,obj.shape[1]//2:]=.08
            a[y:y+obj.shape[0],x:x+obj.shape[1]]=obj
        if name=='distractors':
            for y,x in [(3,5),(14,49)]:a[y:y+9,x:x+11]=q+.17*np.random.default_rng(y+x).normal(size=q.shape)
        if name=='brightness_decoy':
            y,x=np.indices(q.shape);a[25:34,8:19]=q+.09*np.cos(.8*x+1.1*y);a[12:21,43:54]=3.
        if name=='low_texture':a[:]=.08;a[18:27,27:38]=q
        sigma=.16 if name=='noise' else 0.
        observed=a+np.random.default_rng(seed+1000).normal(0,sigma,shape)
        result[name]=dict(clean=a,image=observed,template=q,locations=locations,seed=seed,sigma=sigma,present=bool(locations),description=name)
    return result


def sweep_truth(kind,value):
    s=scene_truth()['unique'];a=s['clean'].copy();t=s['template'];y,x=s['locations'][0]
    if kind=='noise':a+=np.random.default_rng(1808).normal(0,value,a.shape)
    elif kind=='occlusion':
        n=int(np.floor(value*t.shape[1]));a[y:y+t.shape[0],x+t.shape[1]-n:x+t.shape[1]]=.08
    elif kind=='contrast':a[y:y+t.shape[0],x:x+t.shape[1]]=value*t+.3
    return a,t,[y,x]
