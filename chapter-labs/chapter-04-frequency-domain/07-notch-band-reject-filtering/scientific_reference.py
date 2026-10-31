"""Straightforward independent audit oracles, not used by filtering algorithms.

These formulas target the finite experiment settings; production uses log-space
profiles and explicit modular orbit evaluation for numeric extremes instead.
"""
import numpy as np


def axes(shape):
    result=[]
    for n in shape:
        k=np.arange(n);result.append(np.where(k<(n+1)//2,k,k-n)/n)
    return result


def radial(shape):
    y,x=axes(shape)
    return np.sqrt(y[:,None]**2+x[None,:]**2)


def notch(shape,centers,radius,kind,order=2):
    y,x=axes(shape);h=np.ones(shape);unique=set()
    for center in centers:
        for sign in [-1,1]:
            c=tuple(-.5 if sign*v in [-.5,.5] else sign*v for v in center)
            unique.add(c)
    for cy,cx in unique:
        dy=np.abs(y-cy);dx=np.abs(x-cx)
        dy=np.where(dy>.5,1-dy,dy);dx=np.where(dx>.5,1-dx,dx)
        d2=dy[:,None]**2+dx[None,:]**2
        if kind=='ideal':factor=(np.sqrt(d2)>radius).astype(float)
        elif kind=='butterworth':factor=d2**order/(d2**order+radius**(2*order))
        else:factor=1-np.exp(-d2/(2*radius**2))
        h=h*factor
    h[0,0]=1.
    return h


def band(shape,center,width,kind='butterworth',order=2):
    d=radial(shape)
    if kind=='ideal':h=(np.abs(d-center)>width/2).astype(float)
    else:
        h=np.ones(shape);valid=d>0;delta=np.abs(d[valid]**2-center**2)
        if kind=='butterworth':h[valid]=delta**(2*order)/(delta**(2*order)+(d[valid]*width)**(2*order))
        else:h[valid]=1-np.exp(-(delta/(d[valid]*width))**2)
    h[0,0]=1.
    return h


def responses(shape,components,radius=.012,order=2):
    centers=[c['frequency_cpp'] for c in components]
    result=dict(noisy=np.ones(shape))
    for kind in ['ideal','butterworth','gaussian']:result[kind]=notch(shape,centers,radius,kind,order)
    rings=np.ones(shape)
    for r in {float(np.hypot(*c)) for c in centers}:rings*=band(shape,r,2*radius,order=order)
    result['radial_butterworth']=rings
    result['lowpass']=np.exp(-radial(shape)**2/(2*.1**2))
    return result


def truth():
    y,x=np.indices((96,128),dtype=float)
    clean=.4+.16*((x>25)&(x<90)&(y>24)&(y<70))+.12*np.exp(-((x-76)/17)**2-((y-44)/13)**2)+.06*np.cos(2*np.pi*18*y/96)+.035*np.sin(2*np.pi*(5*x/128+3*y/96))
    definitions={
        'horizontal':[(16/96,0,.18,.3)],'vertical':[(0,24/128,.18,.3)],
        'diagonal':[(12/96,20/128,.18,.3)],
        'multiple':[(16/96,0,.14,.3),(0,24/128,.16,.7),(9/96,-17/128,.12,1.1)],
        'high':[(0,24/128,.65,.3)],'weak':[(0,24/128,.015,.3)],
        'leakage':[(10.35/96,22.4/128,.18,.3)],'near_nyquist':[(0,63/128,.18,.3)],
        'nyquist':[(0,.5,.18,0.)],'gaussian_mixed':[(0,24/128,.18,.3),(7/96,15/128,.12,.8)]}
    scenes={}
    for name,parts in definitions.items():
        periodic=sum(a*np.cos(2*np.pi*(fy*y+fx*x)+phase) for fy,fx,a,phase in parts)
        gaussian=np.random.default_rng(707).normal(0,.025,clean.shape) if name=='gaussian_mixed' else np.zeros(clean.shape)
        components=[dict(frequency_cpp=[fy,fx],amplitude=a,phase=p) for fy,fx,a,p in parts]
        scenes[name]=dict(clean=clean,noisy=clean+periodic+gaussian,periodic=periodic,gaussian=gaussian,components=components,gaussian_sigma=.025 if name=='gaussian_mixed' else 0.,seed=707)
    return scenes


def gradient(a):
    return (np.concatenate((a[1:],a[:1]),axis=0)-a,
            np.concatenate((a[:,1:],a[:,:1]),axis=1)-a)


def quality(scene,h):
    clean=scene['clean'];p=scene['periodic'];fc=np.fft.fft2(clean);fp=np.fft.fft2(p)
    inverse=np.fft.ifft2(np.fft.fft2(scene['noisy'])*h);a=inverse.real
    c=np.fft.ifft2(fc*h).real;q=np.fft.ifft2(fp*h).real
    y,x=axes(a.shape);mask=np.zeros(a.shape,dtype=bool)
    for component in scene['components']:
        for sign in [-1,1]:
            fy,fx=np.asarray(component['frequency_cpp'])*sign
            dy=np.abs(y-fy);dx=np.abs(x-fx)
            mask|=(np.minimum(dy,1-dy)[:,None]**2+np.minimum(dx,1-dx)[None,:]**2)<=.02**2
    gain=float(np.linalg.norm((fp*h)[mask])/np.linalg.norm(fp[mask]))
    outside=(~mask)&(radial(a.shape)>.025)
    off=float(np.linalg.norm((fc*h-fc)[outside])/np.linalg.norm(fc[outside]))
    dy,dx=gradient(clean);ay,ax=gradient(a);cy,cx=gradient(c)
    mse=float(np.mean((a-clean)**2))
    return dict(mse=mse,psnr_db=float(10*np.log10(1/mse)) if mse else None,psnr_infinite=bool(mse==0),
                noise_spectral_gain=gain,attenuation_db=float(20*np.log10(gain)) if gain else None,
                periodic_rms_gain=float(np.linalg.norm(q)/np.linalg.norm(p)),off_noise_clean_spectral_error=off,
                clean_gradient_ratio=float(np.sqrt(np.mean(cy**2+cx**2))/np.sqrt(np.mean(dy**2+dx**2))),
                gradient_error_rmse=float(np.sqrt(np.mean((ay-dy)**2+(ax-dx)**2))),
                imaginary_residual=float(np.max(abs(inverse.imag))))


def ringing(h):
    m,n=h.shape;step=np.tile(np.arange(n)>=n//2,(m,1))
    a=np.fft.ifft2(np.fft.fft2(step)*h).real;k=np.fft.ifft2(h).real
    y,x=np.indices(h.shape);distance=np.sqrt(np.minimum(y,m-y)**2+np.minimum(x,n-x)**2)
    return dict(step_overshoot=float(max(0,a.max()-1)),step_undershoot=float(max(0,-a.min())),
                impulse_tail_energy_ratio=float((k[distance>4]**2).sum()/(k**2).sum()))


def proposals(image,max_pairs=3):
    """Reference ranking using a stacked neighborhood maximum and sorted list."""
    mag=abs(np.fft.fft2(image));logs=np.log1p(mag);fy,fx=axes(image.shape)
    allowed=radial(image.shape)>.035;sample=logs[allowed]
    median=np.median(sample);mad=np.median(abs(sample-median));scores=(logs-median)/(1.4826*mad+np.finfo(float).eps)
    neighborhood=np.max(np.stack([np.roll(mag,(i,j),axis=(0,1)) for i in [-1,0,1] for j in [-1,0,1]]),axis=0)
    numerical_floor=64*np.finfo(float).eps*max(1.,float(mag.max()))
    candidates=[(int(r),int(c)) for r,c in zip(*np.where(allowed&(scores>=6)&(mag>=neighborhood)&(mag>numerical_floor)))]
    candidates.sort(key=lambda p:(-float(mag[p]),p[0],p[1]));chosen=[];seen=set();m,n=image.shape
    for r,c in candidates:
        rep=min((r,c),((-r)%m,(-c)%n))
        if rep in seen:continue
        seen.add(rep);f=np.array([fy[r],fx[c]])
        distances=[]
        for a,b in chosen:
            for sign in [-1,1]:
                delta=np.abs(f-sign*np.array([fy[a],fx[b]]));delta=np.minimum(delta,1-delta);distances.append(np.sqrt(np.sum(delta**2)))
        if distances and min(distances)<.025:continue
        chosen.append(rep)
        if len(chosen)==max_pairs:break
    return [dict(index=[r,c],partner_index=[(-r)%m,(-c)%n],frequency_cpp=[float(fy[r]),float(fx[c])],
                 partner_frequency_cpp=[float(fy[(-r)%m]),float(fx[(-c)%n])],magnitude=float(mag[r,c]),contrast_score=float(scores[r,c])) for r,c in chosen]
