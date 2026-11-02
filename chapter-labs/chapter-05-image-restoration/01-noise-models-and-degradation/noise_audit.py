"""Standalone independent scientific validation plus portable artifact integrity."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from noise_support import portable
from noise_reference import moments, uncertainty, spatial, summary, distribution

FIGURES='degradation models pdfs gaussian_sweep impulse_sweep positive_distributions estimation spectra periodic blur equal_variance summary'.split()
TABLES=['models','sweeps','impulses','patches']
EXPECTED={n+'.png' for n in FIGURES}|{n+'.csv' for n in TABLES}|{'metrics.json','reference_arrays.npz'}


def close(actual, expected, label, atol=1e-12, rtol=1e-10):
    if expected is None:
        if actual is not None:
            raise ValueError('Expected null: '+label)
    elif isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):
            raise ValueError('Field mismatch: '+label)
        for key in expected:
            close(actual[key],expected[key],label+'.'+key,atol,rtol)
    elif isinstance(expected,(str,bool)):
        if actual!=expected:
            raise ValueError('Label mismatch: '+label)
    elif not np.allclose(actual,expected,rtol=rtol,atol=atol):
        raise ValueError('Scientific mismatch: '+label)


def audit(directory):
    directory=Path(directory)
    def read_json(path):
        def reject(value):
            raise ValueError('Nonfinite JSON '+value)
        return json.loads(portable.canonical(path),parse_constant=reject)
    entries=read_json(directory/'artifact_manifest.json')['files']
    if len(entries)!=len(EXPECTED) or {e['path'] for e in entries}!=EXPECTED:
        raise ValueError('Manifest inventory mismatch')
    if {p.name for p in directory.iterdir() if p.is_file()}!=EXPECTED|{'artifact_manifest.json'}:
        raise ValueError('Unexpected/missing output files')
    normalized=[]
    for entry in entries:
        path=directory/entry['path'];raw=path.read_bytes();data=portable.canonical(path)
        policy='utf8-crlf-to-lf' if path.suffix in ('.csv','.json') else 'raw-binary'
        if len(data)!=entry['canonical_bytes'] or portable.digest(data)!=entry['sha256'] or policy!=entry['policy']:
            raise ValueError('Integrity failure: '+path.name)
        if raw!=data:
            normalized.append(path.name)
        if path.suffix=='.png':
            with Image.open(path) as im:
                if im.format!='PNG':
                    raise ValueError('Expected PNG encoding')
                im.verify()
            with Image.open(path) as im:
                im.load()
    with np.load(directory/'reference_arrays.npz',allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    if any(v.dtype.kind not in 'fciu' or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Array type/finiteness failure')
    metrics=read_json(directory/'metrics.json');shape=(96,128);n=96*128
    if set(metrics)!={'settings','models','sweeps','impulses','periodic','patches','degradation'}:
        raise ValueError('Metrics inventory mismatch')
    close(metrics['settings'],dict(seed=501,shape=[96,128],data_range=1.,ddof_population=0,ddof_sample=1,
                                  boundary='periodic',psf_anchor=[1,1],display_range=[0,1]),'settings',0,0)
    # Independent geometry: literal predicates and additive masks, not scene API.
    yy,xx=np.indices(shape);yn,xn=yy/95,xx/127
    clean=.15+.18*xn+.08*yn
    clean+=.28*((xn>.12)&(xn<.36)&(yn>.2)&(yn<.65))
    clean+=.3*((xn-.7)**2+(yn-.38)**2<.12**2)
    clean+=np.where((xn>.5)&(yn>.65),.025*np.sin(2*np.pi*.16*xx),0)
    clean[8:28,8:28]=.28
    psf=np.array([[0.,1.,2.],[3.,7.,1.],[2.,0.,0.]])/16
    expected={'clean':clean,'psf':psf};variance=.0016
    parameters={'gaussian':(0.,.04),'rayleigh':(0.,4*variance/(4-math.pi)),
                'erlang':(math.sqrt(3/variance),3.),'exponential':(1/math.sqrt(variance),1.),
                'uniform':(-math.sqrt(3*variance),math.sqrt(3*variance))}
    if set(metrics['models'])!=set(parameters):
        raise ValueError('Model inventory mismatch')

    def draw(name,a,b,size,seed):
        rng=np.random.default_rng(seed)
        if name=='gaussian':return rng.normal(a,b,size)
        if name=='rayleigh':return a+rng.rayleigh(math.sqrt(b)/math.sqrt(2),size)
        if name=='erlang':return rng.gamma(b,1/a,size)
        if name=='exponential':return rng.exponential(1/a,size)
        return rng.uniform(a,b,size)

    def quality_ref(original,observed):
        mse=float(np.mean((original-observed)**2))
        return dict(mse=mse,psnr_db=None if mse==0 else 10*math.log10(1/mse),psnr_status='exact' if mse==0 else 'finite')

    for index,(name,(a,b)) in enumerate(parameters.items()):
        noise=draw(name,a,b,shape,501+index);observed=clean+noise
        sample=draw(name,a,b,(256,256),1501+index)
        edges=np.linspace(float(sample.min()),float(sample.max()),65)
        counts,_=np.histogram(sample,edges);density=counts/(sample.size*np.diff(edges));values=np.sort(sample.ravel())
        power=abs(np.fft.fft2(noise))**2/n
        parts=dict(noise=noise,observed=observed,sample=sample,edges=edges,counts=counts,density=density,
                   ecdf_values=values,ecdf_p=np.arange(1,sample.size+1)/sample.size,
                   quantiles=np.quantile(sample,[.01,.25,.5,.75,.99],method='linear'),
                   power=power,fy=np.fft.fftfreq(96),fx=np.fft.fftfreq(128),directional_y=power.sum(axis=1),directional_x=power.sum(axis=0))
        expected.update({name+'_'+key:value for key,value in parts.items()})
        mu,var,_=moments(name,a,b);measured=summary(sample);se_mean,se_var=uncertainty(name,a,b,sample.size)
        target=dict(a=a,b=b,seed=501+index,sample_seed=1501+index,theory_mean=mu,theory_variance=var,
                    sample_statistics=measured,noise_statistics=summary(noise),observed_statistics=summary(observed),
                    quality=quality_ref(clean,observed),mean_standard_error=se_mean,variance_standard_error=se_var,
                    mean_error_SE=(measured['mean']-mu)/se_mean,variance_error_SE=(measured['sample_variance']-var)/se_var)
        close(metrics['models'][name],target,name)
        if abs(target['mean_error_SE'])>6 or abs(target['variance_error_SE'])>6:
            raise ValueError('Sample outside declared six-SE diagnostic')
        cdf=distribution(name,a,b).cdf(values);ranks=np.arange(1,sample.size+1)/sample.size
        ks=max(float(np.max(ranks-cdf)),float(np.max(cdf-(ranks-1/sample.size))))
        if ks>math.sqrt(math.log(2/1e-6)/(2*sample.size)):
            raise ValueError('Empirical CDF outside DKW bound')
    if [len(metrics[k]) for k in ['sweeps','impulses','periodic','patches']]!=[6,6,6,2]:
        raise ValueError('Experiment row inventory mismatch')
    for k,sigma in enumerate([0.,.01,.025,.05,.1,.2]):
        noise=np.random.default_rng(2501).normal(0,sigma,shape)
        expected[f'gaussian_sweep_{k}']=noise
        close(metrics['sweeps'][k],dict(sigma=sigma,**summary(noise),**quality_ref(clean,clean+noise)),'sweep')
    for k,(pp,ps) in enumerate([(0.,0.),(0.,.08),(.08,0.),(.04,.04),(.1,.1),(.25,.25)]):
        u=np.random.default_rng(3501).random(shape)
        labels=np.where(u<pp,-1,np.where(u<pp+ps,1,0)).astype(np.int8)
        observed=np.where(labels==-1,0,np.where(labels==1,1,clean))
        expected.update({f'impulse_{k}':observed,f'labels_{k}':labels,f'impulse_residual_{k}':observed-clean})
        close(metrics['impulses'][k],dict(p_pepper=pp,p_salt=ps,seed=3501,pepper_fraction=float(np.mean(labels==-1)),
                                        salt_fraction=float(np.mean(labels==1)),**quality_ref(clean,observed)),'impulse')
    specs=[('horizontal',[(.125,0,.08,.3)]),('vertical',[(0,.15625,.08,.3)]),
           ('diagonal',[(.125,.15625,.08,.3)]),('multiple',[(.125,0,.06,.3),(0,.15625,.06,.7)]),
           ('leakage',[(12.4/96,20.3/128,.08,.3)]),('near_nyquist',[(0,63/128,.08,.3)])]
    for row,(name,components) in zip(metrics['periodic'],specs):
        if set(row)!={'name','components','top_bins','nearest_pair_power_fraction','statistics'}:
            raise ValueError('Periodic metadata fields')
        eta=sum(amp*np.cos(2*np.pi*(fy*yy+fx*xx)+phase) for fy,fx,amp,phase in components)
        power=abs(np.fft.fft2(eta))**2/n
        expected[name+'_periodic'],expected[name+'_periodic_power']=eta,power
        close(row['name'],name,'periodic name')
        target=[dict(fy=fy,fx=fx,amplitude=amp,phase=phase,wavevector_angle_degrees=math.degrees(math.atan2(fy,fx))) for fy,fx,amp,phase in components]
        if len(row['components'])!=len(target):raise ValueError('Periodic component count')
        for c,t in zip(row['components'],target):close(c,t,'component')
        pairs=set()
        for fy,fx,_,_ in components:
            k,l=int(np.rint(fy*96))%96,int(np.rint(fx*128))%128
            pairs.update([(k,l),((-k)%96,(-l)%128)])
        close(row['nearest_pair_power_fraction'],sum(power[k,l] for k,l in pairs)/power.sum(),'pair power fraction')
        close(row['statistics'],summary(eta),'periodic statistics')
        bins=row['top_bins']
        if len(bins)!=6 or len({tuple(i) for i in bins})!=6 or any(not 0<=k<96 or not 0<=l<128 for k,l in bins):
            raise ValueError('Invalid peak indices')
        actual_power=arrays[name+'_periodic_power']
        selected=[actual_power[k,l] for k,l in bins]
        close(selected,np.sort(actual_power.ravel())[-6:][::-1],'peak powers',atol=1e-20)
    blur=spatial(clean,psf);zero=spatial(clean,psf,'zero')
    noise=np.random.default_rng(4501).normal(0,.04,shape)
    expected.update(blur_periodic=blur,blur_zero=zero,blur_noise=noise,blur_observed=blur+noise,identity_observed=clean.copy())
    origin=np.zeros(shape)
    for i,j in np.ndindex(psf.shape):origin[(i-1)%96,(j-1)%128]+=psf[i,j]
    expected['psf_origin']=origin
    expected['fourier_observed']=np.fft.fft2(blur+noise)
    expected['fourier_model']=np.fft.fft2(origin)*np.fft.fft2(clean)+np.fft.fft2(noise)
    degradation=metrics['degradation']
    if set(degradation)!={'identity_max_error','spatial_max_error','fourier_max_error','blur_only_quality','noise_only_quality','combined_quality'}:
        raise ValueError('Degradation metadata fields')
    for key in ['identity_max_error','spatial_max_error','fourier_max_error']:
        residual=degradation[key]
        if not isinstance(residual,(float,int)) or not 0<=residual<=1e-10:
            raise ValueError('Roundoff residual exceeds declared bound')
    for key,observed in [('blur_only_quality',blur),('noise_only_quality',clean+noise),('combined_quality',blur+noise)]:
        close(degradation[key],quality_ref(clean,observed),'degradation quality')
    for row,(name,ys,xs) in zip(metrics['patches'],[('flat',(8,28),(8,28)),('textured',(65,90),(70,120))]):
        sl=(slice(*ys),slice(*xs));c=clean[sl];o=expected['gaussian_observed'][sl];eta=expected['gaussian_noise'][sl]
        close(row,dict(name=name,rows=list(ys),columns=list(xs),clean_variance=summary(c)['variance'],
                       observed_variance=summary(o)['variance'],true_noise_variance=summary(eta)['variance'],
                       mean_noise_estimate=summary(o)['mean']-summary(c)['mean']),'patch estimate')
    if set(arrays)!=set(expected):raise ValueError('NPZ key inventory mismatch')
    for key,truth in expected.items():
        actual=arrays[key]
        if actual.shape!=truth.shape or actual.dtype!=truth.dtype:
            raise ValueError('Shape/dtype mismatch: '+key)
        if truth.dtype.kind in 'iu':
            if not np.array_equal(actual,truth):raise ValueError('Integer array mismatch: '+key)
        else:
            close(actual,truth,'array '+key,atol=1e-20 if 'power' in key or 'directional_' in key else 1e-12)
    targets={'models':[dict(model=name,**{k:v for k,v in row.items() if not isinstance(v,dict)},
                            **{'sample_'+k:v for k,v in row['sample_statistics'].items()},**row['quality']) for name,row in metrics['models'].items()],
             'sweeps':metrics['sweeps'],'impulses':metrics['impulses'],'patches':metrics['patches']}
    for name,rows in targets.items():
        parsed=list(csv.DictReader(io.StringIO(portable.canonical(directory/(name+'.csv')).decode('utf-8'))))
        if len(parsed)!=len(rows):raise ValueError('CSV row count mismatch')
        for row,target in zip(parsed,rows):
            if set(row)!=set(target):raise ValueError('CSV column mismatch')
            for key,value in target.items():
                if value is None:
                    if row[key]!='':raise ValueError('CSV null mismatch')
                elif isinstance(value,str):close(row[key],value,'CSV label')
                elif isinstance(value,list):close(json.loads(row[key]),value,'CSV coordinates',0,0)
                else:close(float(row[key]),value,'CSV value')
    return dict(figures=len(FIGURES),arrays=len(arrays),tables={k:len(v) for k,v in targets.items()},
                normalized_crlf=normalized,scientific_checks='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',nargs='?',type=Path,default=Path(__file__).resolve().parent/'output')
    print(json.dumps(audit(parser.parse_args().directory),indent=2))
