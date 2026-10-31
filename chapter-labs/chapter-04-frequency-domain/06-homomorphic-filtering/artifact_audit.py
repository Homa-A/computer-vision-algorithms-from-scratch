"""Independent formula/content audit, including portable canonical text hashes."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from portable_artifacts import canonical, canonical_text, digest

FIGURES = ('decomposition pipeline response gamma_sweep transition_sweep corrections '
           'dark_detail baselines noise_sensitivity boundaries quantitative').split()
EXPECTED = {n+'.png' for n in FIGURES} | {'measurements.csv','parameters.csv','noise.csv',
            'radial.csv','metrics.json','reference_arrays.npz'}


def close(a,b,label):
    if not np.allclose(a,b,rtol=1e-10,atol=1e-10):
        raise ValueError(f'Scientific mismatch: {label}; max error={np.max(np.abs(np.asarray(a)-b))}')


def radius(shape):
    # Independent modular signed integer indexing, including odd lengths.
    axes=[]
    for n in shape:
        k=np.arange(n); axes.append(np.where(k < (n+1)//2,k,k-n)/n)
    return np.sqrt(axes[0][:,None]**2+axes[1][None,:]**2)


def response(shape,lo=.6,hi=1.4,cutoff=.06,c=1.):
    return lo+(hi-lo)*(1-np.exp(-c*radius(shape)**2/cutoff**2))


def transform(a,lo=.6,hi=1.4,cutoff=.06,c=1.,floor=None,pad=0):
    if floor is not None:a=np.maximum(a,floor)
    a=np.pad(a,((pad,pad),(pad,pad)),mode='reflect')
    z=np.fft.ifft2(np.fft.fft2(np.log(a))*response(a.shape,lo,hi,cutoff,c))
    shape=(a.shape[0]-2*pad,a.shape[1]-2*pad)
    return np.exp(z.real)[pad:pad+shape[0],pad:pad+shape[1]],float(np.max(abs(z.imag)))


def independent_metrics(a,b):
    la,lb=np.log(a),np.log(b);offset=float((la-lb).mean()); residual=la-lb-offset
    d=radius(a.shape)
    low=np.fft.ifft2(np.fft.fft2(residual)*np.exp(-d*d/(2*.03**2))).real
    hp=1-np.exp(-d*d/(2*.08**2))
    sa=np.fft.ifft2(np.fft.fft2(la)*hp).real.std();sb=np.fft.ifft2(np.fft.fft2(lb)*hp).real.std()
    return dict(log_scale_offset=offset,log_reflectance_rmse=float(np.sqrt(np.mean(residual**2))),
                low_frequency_log_residual_std=float(low.std()),detail_std_ratio=float(sa/sb) if sb>1e-14 else None,
                aligned_intensity_rmse=float(np.sqrt(np.mean((np.exp(la-offset)-b)**2))),log_dynamic_range=float(np.ptp(la)),mean_intensity=float(a.mean()))


def audit(directory):
    directory=Path(directory)
    manifest=json.loads(canonical_text((directory/'artifact_manifest.json').read_bytes()))
    entries=manifest['files']
    if len(entries)!=len(EXPECTED) or {e['path'] for e in entries}!=EXPECTED:
        raise ValueError('Unexpected or missing scientific files.')
    reports=[]
    for e in entries:
        p=directory/e['path'];raw=p.read_bytes();data=canonical(p)
        policy='utf8-crlf-to-lf' if p.suffix in ('.json','.csv') else 'raw-binary'
        if e['policy']!=policy or len(data)!=e['canonical_bytes'] or digest(data)!=e['sha256']:
            raise ValueError('Hash/content integrity failure: '+p.name)
        reports.append(dict(path=p.name,physical_bytes=len(raw),canonical_bytes=len(data),raw_sha256=digest(raw),canonical_sha256=digest(data),normalized_crlf=raw!=data))
        if p.suffix=='.png':
            with Image.open(p) as image:
                if image.format!='PNG':raise ValueError('Expected PNG')
                image.verify()
            with Image.open(p) as image:image.load()
    metrics=json.loads(canonical(directory/'metrics.json'))
    tables={}
    for name,count in [('measurements',25),('parameters',15),('noise',3),('radial',1503)]:
        rows=list(csv.DictReader(io.StringIO(canonical(directory/(name+'.csv')).decode('utf-8'))))
        if len(rows)!=count:raise ValueError('Wrong row count: '+name)
        tables[name]=rows
        if name!='radial':
            for row,expected in zip(rows,metrics[name]):
                if set(row)!=set(expected):raise ValueError('Table columns differ')
                for k,v in expected.items():
                    if isinstance(v,str):
                        if row[k]!=v:raise ValueError('Label mismatch')
                    elif v is None:
                        if row[k]!='':raise ValueError('Null mismatch')
                    else:close(float(row[k]),v,name+'/'+k)
    with np.load(directory/'reference_arrays.npz',allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    scene_names=['uniform','gradient','vignette','sinusoidal','dark']
    keys={'reflectance','baseline_tone','baseline_lowpass','baseline_identity','log_input','log_spectrum','response','filtered_log','out_reflect','noise_bright','out_noisy_bright'}
    keys|={f'{p}_{n}' for p in ['illum','observed','out'] for n in scene_names}
    keys|={f'{p}_{j}' for p in ['raw_sensor','noise_output'] for j in range(3)}
    if set(arrays)!=keys:raise ValueError('Array inventory mismatch')
    for k,a in arrays.items():
        if a.shape!=(96,128) or a.dtype!=('complex128' if k=='log_spectrum' else 'float64') or not np.isfinite(a).all():
            raise ValueError('Invalid reference array '+k)
    y,x=np.indices((96,128),dtype=float);y/=96;x/=128
    truth=np.exp(.12*np.cos(2*np.pi*(17*x+5*y))+.08*np.sin(2*np.pi*31*x)+.18*((x>.22)&(x<.72)&(y>.23)&(y<.75)))
    close(arrays['reflectance'],truth,'analytic reflectance')
    illum=dict(uniform=np.full(truth.shape,.7),gradient=np.exp(-1.3+1.6*x),vignette=np.exp(-2.2*((x-.5)**2+(y-.5)**2)),sinusoidal=np.exp(-.6+.7*np.cos(2*np.pi*x)+.35*np.sin(2*np.pi*y)),dark=np.exp(-2.5+1.8*x))
    recomputed=[]
    for name in scene_names:
        f=truth*illum[name];out,imag=transform(f)
        close(arrays['illum_'+name],illum[name],name+' illumination');close(arrays['observed_'+name],f,name+' product');close(arrays['out_'+name],out,name+' correction')
        low=np.fft.ifft2(np.fft.fft2(f)*np.exp(-radius(f.shape)**2/(2*.06**2))).real
        for method,a in dict(input=f,global_power=f**.6,ordinary_lowpass=low,identity=f,homomorphic=out).items():
            row=next(r for r in metrics['measurements'] if r['scene']==name and r['method']==method)
            for key,value in independent_metrics(a,truth).items():
                if value is None:
                    if row[key] is not None:raise ValueError('Undefined detail metric')
                else:close(row[key],value,name+'/'+method+'/'+key)
            if method=='homomorphic':close(row['imaginary_residual'],imag,'imaginary residual')
        if name=='gradient':
            close(arrays['baseline_lowpass'],low,'LP baseline');close(arrays['baseline_tone'],f**.6,'power baseline');close(arrays['baseline_identity'],f,'identity baseline')
    f=arrays['observed_gradient']
    close(arrays['log_input'],np.log(f),'log input');close(arrays['log_spectrum'],np.fft.fft2(np.log(f)),'log FFT');close(arrays['response'],response(f.shape),'H');close(arrays['filtered_log'],np.log(arrays['out_gradient']),'filtered log')
    close(arrays['out_reflect'],transform(f,pad=32)[0],'reflection/crop')
    close(metrics['boundary_max_difference'],np.max(abs(arrays['out_reflect']-arrays['out_gradient'])),'boundary difference')
    for row in metrics['parameters']:
        a,imag=transform(f,row['gamma_low'],row['gamma_high'],row['cutoff'],row['c'])
        for k,v in independent_metrics(a,truth).items():
            if v is not None:close(row[k],v,'parameter '+k)
        close(row['imaginary_residual'],imag,'parameter imaginary')
    for row in tables['radial']:
        d,c=float(row['radius_cpp']),float(row['c'])
        close(float(row['gain']),.6+.8*(1-np.exp(-c*d*d/.06**2)),'radial gain')
    noise=np.random.default_rng(606).normal(size=truth.shape)
    close(arrays['noise_bright'],arrays['observed_dark']+.004*noise,'bright noise')
    close(arrays['out_noisy_bright'],transform(arrays['noise_bright'])[0],'bright noisy output')
    for j,row in enumerate(metrics['noise']):
        scale=[1.,1e-4,1e-5][j];close(row['scale'],scale,'noise scale');close(row['floor'],1e-7,'noise floor')
        clean=arrays['observed_dark']*scale;raw=clean+2e-6*noise
        a,imag=transform(raw,hi=1.8,floor=1e-7);b,_=transform(clean,hi=1.8,floor=1e-7)
        close(arrays[f'raw_sensor_{j}'],raw,'raw sensor');close(arrays[f'noise_output_{j}'],a,'noisy output')
        rms=np.sqrt(np.mean((raw-clean)**2));delta=np.sqrt(np.mean((a-b)**2))
        for key,val in dict(raw_noise_rms=rms,output_delta_rms=delta,noise_gain=delta/rms,log_delta_rms=np.sqrt(np.mean((np.log(a)-np.log(b))**2)),floored_pixels=np.count_nonzero(raw<1e-7),imaginary_residual=imag).items():close(row[key],val,'noise '+key)
    if len(metrics['checks'])!=15 or max(r['max_absolute_error'] for r in metrics['checks'])>1e-10:
        raise ValueError('Analytic check failure')
    return dict(figures=11,arrays=len(arrays),tables={n:len(r) for n,r in tables.items()},files=reports)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',nargs='?',type=Path,default=Path(__file__).resolve().parent/'output')
    print(json.dumps(audit(parser.parse_args().directory),indent=2))
