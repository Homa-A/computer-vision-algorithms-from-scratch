"""Synthetic ground truth; randomness delegates to published Lab 01 unchanged."""
import numpy as np
from mean_support import noise

SHAPE=(64,80)
METHODS=['arithmetic','geometric','harmonic','contra']
Q_VALUES=[-1.5,-1.,-.5,0.,.5,1.,1.5]
WINDOWS=[(1,1),(1,3),(3,1),(3,3),(5,5),(7,7),(3,5)]
IMPULSE_SCENES=['salt','pepper','mixed','black_salt','black_pepper','black_mixed']
DETAILS=dict(dark_line=dict(row=36,columns=[6,32],profile_column=20,profile_rows=[31,42]),
             bright_line=dict(row=44,columns=[46,74],profile_column=60,profile_rows=[39,50]),
             bright_point=[28,15],dark_point=[28,25],step_column=40)


def clean_image():
    y,x=np.indices(SHAPE,dtype=float)
    image=.32+.10*x/79+.04*y/63
    image[:,40:]+=.26
    image[8:23,6:20]+=.13
    image[36,6:32]=.12
    image[44,46:74]=.92
    image[28,15]=.95;image[28,25]=.08
    image[48:61,5:33]=.45+.04*np.sin(2*np.pi*x[48:61,5:33]/3)
    image[8:24,50:70]+=.035*((x[8:24,50:70]+y[8:24,50:70])%2)
    return image


def make_scenes():
    positive=clean_image();black=positive.copy()
    black[36,6:32]=0;black[44,46:74]=1.;black[28,15]=1.;black[28,25]=0.
    result={}
    configs=[('gaussian','gaussian',551),('uniform','uniform',552),('salt','salt',553),
             ('pepper','pepper',554),('mixed','mixed',555),('black_salt','salt',556),
             ('black_pepper','pepper',557),('black_mixed','mixed',558)]
    for name,kind,seed in configs:
        clean=black if name.startswith('black_') else positive
        rng=np.random.default_rng(seed)
        if kind in ['gaussian','uniform']:
            law=noise.Distribution(kind,0.,.025) if kind=='gaussian' else noise.Distribution(kind,-.05,.05)
            eta=law.sample(SHAPE,rng=rng);observed=clean+eta
            labels=np.zeros(SHAPE,np.int8);mask=np.ones(SHAPE,bool)
            parameters=dict(kind=kind,a=law.a,b=law.b)
        else:
            pp=.08 if kind=='pepper' else (.04 if kind=='mixed' else 0.)
            ps=.08 if kind=='salt' else (.04 if kind=='mixed' else 0.)
            observed,labels=noise.impulse_replace(clean,p_pepper=pp,p_salt=ps,rng=rng)
            eta=observed-clean;mask=labels!=0
            parameters=dict(kind=kind,p_pepper=pp,p_salt=ps,pepper=0.,salt=1.)
        if (observed<0).any():raise ValueError('Positive-domain experiment has a negative draw; no clipping is allowed')
        result[name]=dict(clean=clean.copy(),observed=observed,noise=eta,labels=labels,mask=mask,seed=seed,parameters=parameters)
    return result


def step_image():
    return np.broadcast_to(np.where(np.arange(80)<40,.25,.75),(64,80)).copy()


STRESS=[('tiny',[1e-300,2e-300,4e-300],1.5),('huge',[1e300,2e300,4e300],1.5),
        ('span',[1e-300,1.,1e300],-.5),('zero',[0.,.5,1.],-1.5),
        ('allzero',[0.,0.,0.],1.5),
        ('near_huge',[1e200*(1-2e-12),1e200*(1-1e-12),1e200],1e12),
        ('positive_limit',[1.,2.,4.],np.finfo(float).max),
        ('negative_limit',[1.,2.,4.],-np.finfo(float).max),
        ('near_max',[np.nextafter(np.finfo(float).max,0),np.finfo(float).max,np.finfo(float).max],-.5)]
