"""Original synthetic scenes, explicit seeds and paired controlled experiments."""
import numpy as np
from order_support import noise

METHODS=('median','minimum','maximum','midpoint','alpha_trimmed','arithmetic')
FIGURES=('overview','median_impulses','min_max','midpoint_uniform','trim_sweep','mean_comparison','details','windows','combined','density_failure','boundaries','quantitative')
GROUPS=('main','trim','windows','density','calibration','boundaries','stress')


def scene():
    y,x=np.indices((40,56));a=.25+.15*x/55+.15*(x>=28)+.1*(y>=20)
    a[3:15,35:51]=np.where(y[3:15,35:51]>x[3:15,35:51]-35,.75,.25)
    a[23,4:24]=.9;a[27:29,4:24]=.9;a[32,4:24]=.1;a[35:37,4:24]=.1
    a[17,10]=.95;a[17,20]=.05
    a[24:37,34:51]=.5+.15*((x[24:37,34:51]+y[24:37,34:51])%2*2-1)
    a[abs(x-25)+abs(y-10)==6]=.92
    return a


def masks():
    out={};y,x=np.indices((40,56))
    for name,sl in [('bright1',(slice(23,24),slice(4,24))),('bright2',(slice(27,29),slice(4,24))),('dark1',(slice(32,33),slice(4,24))),('dark2',(slice(35,37),slice(4,24))),('texture',(slice(24,37),slice(34,51)))]:
        a=np.zeros((40,56),bool);a[sl]=True;out[name]=a
    out['ring']=abs(x-25)+abs(y-10)==6
    out['objects']=np.zeros((40,56),bool);out['objects'][17,[10,20]]=True
    return out


def observation(case):
    a=case['clean'].copy();kind=case['kind'];rng=np.random.default_rng(case['seed']);sd=case['sigma']
    labels=np.zeros(a.shape,np.int8)
    if kind=='border':
        a[0,[0,-1]]=1.;labels[0,[0,-1]]=1;a[-1,[0,-1]]=0.;labels[-1,[0,-1]]=-1
    else:
        if kind in ('gaussian','gaussian_impulse'):a+=noise.Distribution('gaussian',0.,sd).sample(a.shape,rng=rng)
        if kind in ('uniform','uniform_impulse') and sd>0:
            b=np.sqrt(3)*sd;a+=noise.Distribution('uniform',-b,b).sample(a.shape,rng=rng)
        if kind in ('salt','pepper','mixed','uniform_impulse','gaussian_impulse'):
            p=case['rate'];pp=p if kind=='pepper' else 0. if kind=='salt' else p/2
            ps=p if kind=='salt' else 0. if kind=='pepper' else p/2
            a,labels=noise.impulse_replace(a,p_pepper=pp,p_salt=ps,rng=rng)
    events=labels!=0
    if kind in ('gaussian','uniform') and sd>0:events=np.ones(a.shape,bool)
    return a,labels,events


def parameters(method,window=(3,3),d=None,mode='reflect'):
    return dict(method=method,window=list(window),d=4 if d is None and method=='alpha_trimmed' else 0 if d is None else d,mode=mode,cval=-.2 if mode=='constant' else 0.)


def design():
    cases=[];tasks=[]
    for i,kind in enumerate(('gaussian','uniform','salt','pepper','mixed','uniform_impulse','gaussian_impulse','high','negative')):
        cases.append(dict(id=kind,kind='mixed' if kind=='high' else 'uniform' if kind=='negative' else kind,seed=730+i,rate=.85 if kind=='high' else .2,sigma=.04,clean=scene()-(1. if kind=='negative' else 0.)))
        tasks.extend((kind,'main',parameters(m)) for m in METHODS)
    for name in ('gaussian','uniform','mixed','uniform_impulse','gaussian_impulse','high'):
        tasks.extend((name,'trim',parameters('alpha_trimmed',d=d)) for d in (0,2,4,6,8))
    cases.append(dict(id='noiseless',kind='uniform',seed=739,rate=0.,sigma=0.,clean=scene()))
    for name in ('mixed','noiseless'):
        for w in (3,5,7):tasks.extend((name,'windows',parameters(m,(w,w),d=(w*w//4*2 if m=='alpha_trimmed' else 0))) for m in METHODS)
    for p in (0.,.1,.3,.5,.7,.9,1.):
        for seed in (750,751,752):
            name=f'density_{p}_{seed}';cases.append(dict(id=name,kind='mixed',seed=seed,rate=p,sigma=0.,clean=np.full((40,56),.5)))
            tasks.extend((name,'density',parameters(m)) for m in ('median','alpha_trimmed','arithmetic'))
    for kind in ('uniform','gaussian'):
        for sd in (.01,.04,.08):
            for seed in (760,761,762,763):
                name=f'flat_{kind}_{sd}_{seed}';cases.append(dict(id=name,kind=kind,seed=seed,rate=0.,sigma=sd,clean=np.full((24,32),.5)))
                tasks.extend((name,'calibration',parameters(m)) for m in ('midpoint','median','arithmetic'))
    border=np.full((12,16),.45);border[0]=.8;border[:,0]=.2;border[-1]=.65
    cases.append(dict(id='border',kind='border',seed=780,rate=0.,sigma=0.,clean=border))
    for mode in ('reflect','edge','constant'):tasks.extend(('border','boundaries',parameters(m,(5,5),mode=mode)) for m in METHODS[:-1])
    return cases,tasks


def stresses():
    mx=np.finfo(float).max;t=np.nextafter(0.,1.)
    return dict(zero=np.zeros((2,3)),negative=np.full((2,3),-2.),large=np.array([[mx,mx/2,mx/3]]),
        cancellation=np.array([[-1e308,1e-300,1e308]]),precision=np.array([[1.,np.nextafter(1.,2.),np.nextafter(np.nextafter(1.,2.),2.)]]),
        ties=np.array([[1.,1.,1.,9.,9.]]),subnormal=np.array([[t,2*t,4*t]]),singleton=np.array([[-5.]]))
