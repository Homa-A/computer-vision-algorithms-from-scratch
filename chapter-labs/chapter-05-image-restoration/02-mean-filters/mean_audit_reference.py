"""Independent full-array spatial oracles at ordinary experimental scales.

Explicit offset enumeration + independent index folding, not pad/stride views.
Decimal and literal scalar oracles live in mean_reference for extreme values.
"""
import math
import numpy as np


def image_truth():
    yy,xx=np.indices((64,80))
    result=.32+.10*xx/79+.04*yy/63
    result+=.26*(xx>=40)
    result+=.13*((yy>=8)&(yy<23)&(xx>=6)&(xx<20))
    result[36,6:32]=.12;result[44,46:74]=.92
    result[28,15]=.95;result[28,25]=.08
    mask=(yy>=48)&(yy<61)&(xx>=5)&(xx<33)
    result[mask]=.45+.04*np.sin(2*np.pi*xx[mask]/3)
    mask=(yy>=8)&(yy<24)&(xx>=50)&(xx<70)
    result[mask]+=.035*((xx[mask]+yy[mask])%2)
    return result


def scene_truth():
    positive=image_truth();black=positive.copy()
    black[36,6:32]=0.;black[44,46:74]=1.;black[28,15]=1.;black[28,25]=0.
    result={}
    configs=[('gaussian','gaussian',551),('uniform','uniform',552),('salt','salt',553),('pepper','pepper',554),('mixed','mixed',555),('black_salt','salt',556),('black_pepper','pepper',557),('black_mixed','mixed',558)]
    for name,kind,seed in configs:
        clean=black if name.startswith('black_') else positive;rng=np.random.default_rng(seed)
        if kind=='gaussian':
            eta=rng.normal(0.,.025,(64,80));observed=clean+eta
            labels=np.zeros((64,80),np.int8);mask=np.ones((64,80),bool);parameters=dict(kind=kind,a=0.,b=.025)
        elif kind=='uniform':
            eta=rng.uniform(-.05,.05,(64,80));observed=clean+eta
            labels=np.zeros((64,80),np.int8);mask=np.ones((64,80),bool);parameters=dict(kind=kind,a=-.05,b=.05)
        else:
            pp=.08 if kind=='pepper' else (.04 if kind=='mixed' else 0.)
            ps=.08 if kind=='salt' else (.04 if kind=='mixed' else 0.)
            u=rng.random((64,80));labels=np.where(u<pp,-1,np.where(u<pp+ps,1,0)).astype(np.int8)
            observed=np.where(labels==-1,0,np.where(labels==1,1,clean));eta=observed-clean;mask=labels!=0
            parameters=dict(kind=kind,p_pepper=pp,p_salt=ps,pepper=0.,salt=1.)
        result[name]=dict(clean=clean,observed=observed,noise=eta,mask=mask,labels=labels,seed=seed,parameters=parameters)
    return result


def full_filter(image,method,window=(3,3),*,Q=1.5,mode='reflect',cval=0.):
    H,W=image.shape;yy,xx=np.indices(image.shape);planes=[]
    for i,j in np.ndindex(window):
        y=yy+i-window[0]//2;x=xx+j-window[1]//2
        outside=(y<0)|(y>=H)|(x<0)|(x>=W)
        if mode=='reflect':
            if H==1:y=np.zeros_like(y)
            else:
                t=y%(2*H-2);y=np.where(t<H,t,2*H-2-t)
            if W==1:x=np.zeros_like(x)
            else:
                t=x%(2*W-2);x=np.where(t<W,t,2*W-2-t)
        else:y=np.clip(y,0,H-1);x=np.clip(x,0,W-1)
        values=image[y,x]
        if mode=='constant':values=np.where(outside,cval,values)
        planes.append(values)
    z=np.stack(planes,axis=0);N=len(planes)
    def finish(value):
        # Exact constant preservation is a mathematical identity, not a floor
        # applied to tiny derived errors/PSNR values.
        return np.where((z==z[:1]).all(axis=0),z[0],value)
    if method=='arithmetic' or (method=='contra' and Q==0):return finish(np.sum(z,axis=0)/N)
    zeros=(z==0).any(axis=0)
    if method=='geometric':return finish(np.prod(z,axis=0)**(1/N))
    if method=='harmonic' or (method=='contra' and Q==-1):
        reciprocals=np.divide(1.,z,out=np.zeros_like(z),where=z>0)
        denominator=reciprocals.sum(axis=0)
        return finish(np.divide(N,denominator,out=np.zeros(image.shape),where=(denominator>0)&(~zeros)))
    numerator=np.power(z,Q+1,out=np.zeros_like(z),where=z>0).sum(axis=0)
    denominator=np.power(z,Q,out=np.zeros_like(z),where=z>0).sum(axis=0)
    return finish(np.divide(numerator,denominator,out=np.zeros(image.shape),where=(denominator>0)&((~zeros) if Q<0 else True)))


def quality_ref(clean,observed,output,mask):
    delta=output-clean;mse=float(np.mean(delta**2));input_mse=float(np.mean((observed-clean)**2))
    return dict(mse=mse,rmse=math.sqrt(mse),mae=float(np.mean(abs(delta))),
                psnr_db=None if mse==0 else -10*math.log10(mse),psnr_status='exact' if mse==0 else 'finite',
                residual_bias=float(np.sum(delta)/delta.size),input_mse=input_mse,residual_ratio=mse/input_mse if input_mse else None,
                mse_corrupted=float(np.mean(delta[mask]**2)) if np.any(mask) else None,
                mse_uncorrupted=float(np.mean(delta[~mask]**2)) if np.any(~mask) else None)


def detail_ref(clean,output):
    dclean=math.fsum(float((clean[35,x]+clean[37,x])/2-clean[36,x]) for x in range(6,32))/26
    bclean=math.fsum(float(clean[44,x]-(clean[43,x]+clean[45,x])/2) for x in range(46,74))/28
    dark=math.fsum(float((output[35,x]+output[37,x])/2-output[36,x]) for x in range(6,32))/26
    bright=math.fsum(float(output[44,x]-(output[43,x]+output[45,x])/2) for x in range(46,74))/28
    dt=(clean[36,20]+(clean[35,20]+clean[37,20])/2)/2
    bt=(clean[44,60]+(clean[43,60]+clean[45,60])/2)/2
    return dict(dark_contrast_ratio=dark/dclean,bright_contrast_ratio=bright/bclean,
                dark_line_width=sum(float(output[y,20])<dt-1e-12 for y in range(31,42)),
                bright_line_width=sum(float(output[y,60])>bt+1e-12 for y in range(39,50)),
                bright_point_error=float(abs(output[28,15]-clean[28,15])),dark_point_error=float(abs(output[28,25]-clean[28,25])))


def edge_ref(image):
    profile=np.sum(image,axis=0)/image.shape[0]
    def cross(level):
        for x in range(1,len(profile)):
            if profile[x-1]<level<=profile[x]:return x-1+(level-profile[x-1])/(profile[x]-profile[x-1])
        raise ValueError('Missing step crossing')
    return dict(edge_width_10_90=float(cross(.7)-cross(.3)),edge_midpoint_shift=float(cross(.5)-39.5),edge_gradient_ratio=float(max(float(profile[x+1]-profile[x]) for x in range(len(profile)-1))/.5))
