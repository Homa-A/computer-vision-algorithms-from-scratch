"""Original deterministic scenes, independent flat ROI and controlled task design."""
import numpy as np
from adaptive_support import noise
FIGURES=('overview','local_statistics','gaussian','variance_estimates','variance_sweep','details','median_comparison','decisions','max_size','high_density','mismatch','quantitative')
GROUPS=('main','variance','windows','density','boundary','calibration','stress')
ROI=(2,10,2,14);BAD_ROI=(12,24,22,32)


def scene():
    y,x=np.indices((32,44));a=.2+.2*x/43+.15*(x>=22)+.1*(y>=16)
    a[1:11,1:15]=.45
    a[3:13,28:41]=np.where(y[3:13,28:41]>x[3:13,28:41]-28,.8,.25)
    a[20,3:19]=1.;a[24,3:19]=0.;a[27:29,3:19]=1.
    a[14,9]=1.;a[14,16]=0.
    a[19:30,27:41]=.5+.15*((x[19:30,27:41]+y[19:30,27:41])%2*2-1)
    return a


def masks():
    m={};shape=(32,44)
    for n,sl in [('flat',(slice(2,10),slice(2,14))),('edge',(slice(12,19),slice(20,24))),('texture',(slice(19,30),slice(27,41))),('bright',(slice(20,21),slice(3,19))),('dark',(slice(24,25),slice(3,19)))]:
        a=np.zeros(shape,bool);a[sl]=True;m[n]=a
    m['objects']=np.zeros(shape,bool);m['objects'][14,[9,16]]=True
    return m


def observe(c):
    clean=c['clean'];a=clean.copy();rng=np.random.default_rng(c['seed']);labels=np.zeros(a.shape,np.int8);kind=c['kind']
    if kind in ('gaussian','nonuniform'):
        sd=c['sigma'];draw=noise.Distribution('gaussian',0.,sd).sample(a.shape,rng=rng)
        if kind=='nonuniform':draw[:,:a.shape[1]//2]=0.
        a+=draw
    elif kind in ('salt','pepper','mixed'):
        p=c['rate'];a,labels=noise.impulse_replace(a,p_pepper=p if kind=='pepper' else 0. if kind=='salt' else p/2,p_salt=p if kind=='salt' else 0. if kind=='pepper' else p/2,rng=rng)
    elif kind=='border':
        a[0,[0,-1]]=1.;labels[0,[0,-1]]=1;a[-1,[0,-1]]=0.;labels[-1,[0,-1]]=-1
    events=labels!=0
    if kind=='gaussian':events=np.ones(a.shape,bool)
    if kind=='nonuniform':events[:,a.shape[1]//2:]=True
    return a,labels,events


def local(window=7,policy='clipped',source='oracle',factor=1.):
    return dict(method='local',window=[window,window],regularization=policy,variance_source=source,factor=float(factor),mode='reflect',cval=0.)


def median(maximum=7,mode='reflect'):
    return dict(method='adaptive_median',initial_size=3,max_size=maximum,mode=mode,cval=-.2 if mode=='constant' else 0.)


def fixed(method,window=7):return dict(method=method,window=[window,window],mode='reflect',cval=0.)


def stress_inputs():
    return dict(zero=np.zeros((2,3)),negative=np.full((2,3),-2.),huge=np.full((2,3),1e308),near=np.array([[1e150,np.nextafter(1e150,np.inf),1e150]]),
                tiny=np.array([[0.,1e-150,2e-150]]),ties=np.array([[0.,0.,1.,1.,.5]]))


def design():
    cases=[];tasks=[]
    for i,(n,kind,sd,rate) in enumerate([('low','gaussian',.015,0.),('medium','gaussian',.05,0.),('high','gaussian',.12,0.),('salt','salt',0.,.2),('pepper','pepper',0.,.2),('mixed','mixed',0.,.2),('severe','mixed',0.,.85),('nonuniform','nonuniform',.08,0.),('clean','clean',0.,0.),('negative','gaussian',.03,0.)]):
        cases.append(dict(id=n,kind=kind,seed=900+i,sigma=sd,rate=rate,clean=scene()-(1. if n=='negative' else 0.)))
        source='assumed_mismatch' if kind in ('salt','pepper','mixed') else 'oracle'
        tasks.extend((n,'main',p) for p in (fixed('mean'),fixed('fixed_median',3),local(source=source),local(policy='literal',source=source),local(source='estimated'),median()))
    for factor in (0.,.25,1.,4.,16.):
        for policy in ('clipped','literal'):tasks.append(('medium','variance',local(policy=policy,factor=factor)))
    tasks.append(('medium','variance',local(source='contaminated')))
    for w in (3,5,9):tasks.extend(('medium','windows',p) for p in (fixed('mean',w),local(w),local(w,source='estimated')))
    for rate in (0.,.1,.3,.5,.7,.9,1.):
        for seed in (950,951,952):
            n=f'density_{rate}_{seed}';cases.append(dict(id=n,kind='mixed',seed=seed,sigma=0.,rate=rate,clean=np.full((24,32),.5)))
            tasks.append((n,'density',fixed('fixed_median',3)))
            tasks.extend((n,'density',median(s)) for s in (3,5,7,9))
    for seed in (970,971,972):
        n=f'flat_{seed}';cases.append(dict(id=n,kind='gaussian',seed=seed,sigma=.05,rate=0.,clean=np.full((24,32),.5)))
        tasks.extend((n,'calibration',p) for p in (fixed('mean'),local(),local(source='estimated'),local(source='contaminated')))
    border=np.full((12,16),.45);border[0]=.8;border[:,0]=.2
    cases.append(dict(id='border',kind='border',seed=980,sigma=.05,rate=0.,clean=border))
    for mode in ('reflect','edge','constant'):
        p=local();p.update(mode=mode,cval=-.2 if mode=='constant' else 0.)
        tasks.extend(('border','boundary',q) for q in (p,median(mode=mode)))
    return cases,tasks
