"""Canonical integrity plus independent scientific/formula validation."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from notch_support import portable
import scientific_reference as reference

FIGURES = ('overview spectral_peaks band_responses notch_responses width_sensitivity '
           'order_sensitivity orientations multiple_noise leakage failure_boundaries '
           'stress_cases quantitative').split()
TABLES=('measurements','parameters','detections','boundaries','radial_profiles')
EXPECTED={n+'.png' for n in FIGURES}|{n+'.csv' for n in TABLES}|{'metrics.json','reference_arrays.npz'}


def close(actual,expected,label):
    if expected is None:
        if actual is not None:raise ValueError('Expected undefined value: '+label)
    elif isinstance(expected,(str,bool)):
        if actual!=expected:raise ValueError('Label/boolean mismatch: '+label)
    elif label.endswith('/attenuation_db') and actual < -260 and expected < -260:
        # An FFT notch can suppress a bin to the float64 roundoff floor.
        # Converting those near-zero gains to dB magnifies insignificant
        # platform-dependent FFT residuals (e.g. -299.103 vs -299.110 dB).
        # Compare linear amplitude gains, not their poorly conditioned logs.
        # Differences above 32 float64 eps remain scientific mismatches.
        actual_gain = 10.0**(actual/20.0)
        expected_gain = 10.0**(expected/20.0)
        if not (np.isfinite(actual_gain) and np.isfinite(expected_gain)
                and np.isclose(actual_gain,expected_gain,rtol=1e-10,
                               atol=32*np.finfo(np.float64).eps)):
            raise ValueError('Scientific mismatch: '+label)
    elif not np.allclose(actual,expected,rtol=1e-10,atol=1e-10):
        raise ValueError('Scientific mismatch: '+label)


def _json(path):
    def invalid(value):raise ValueError('Nonfinite JSON constant '+value)
    return json.loads(portable.canonical(path),parse_constant=invalid)


def audit(directory):
    directory=Path(directory);manifest=_json(directory/'artifact_manifest.json')
    entries=manifest['files']
    if len(entries)!=len(EXPECTED) or {r['path'] for r in entries}!=EXPECTED:
        raise ValueError('Scientific artifact inventory mismatch.')
    actual_files={p.name for p in directory.iterdir() if p.is_file()}
    if actual_files!=EXPECTED|{'artifact_manifest.json'}:raise ValueError('Unexpected/missing output files.')
    file_reports=[]
    for row in entries:
        p=directory/row['path'];raw=p.read_bytes();data=portable.canonical(p)
        policy='utf8-crlf-to-lf' if p.suffix in ('.csv','.json') else 'raw-binary'
        if row['policy']!=policy or row['canonical_bytes']!=len(data) or row['sha256']!=portable.digest(data):
            raise ValueError('Integrity failure: '+p.name)
        file_reports.append(dict(path=p.name,physical_bytes=len(raw),canonical_bytes=len(data),raw_sha256=portable.digest(raw),canonical_sha256=portable.digest(data),normalized_crlf=(raw!=data)))
        if p.suffix=='.png':
            with Image.open(p) as image:
                if image.format!='PNG':raise ValueError('Expected PNG')
                image.verify()
            with Image.open(p) as image:image.load()
    metrics=_json(directory/'metrics.json')
    close(metrics['settings']['shape'],[96,128],'shape');close(metrics['settings']['radius_cpp'],.012,'radius')
    close(metrics['settings']['order'],2,'order');close(metrics['settings']['baseline_cutoff_cpp'],.1,'baseline cutoff')
    close(metrics['settings']['seed'],707,'seed')
    if metrics['settings']['detector']!=dict(dc_radius=.035,threshold=6.,max_pairs=3,separation=.025):raise ValueError('Detector settings differ')
    parsed={}
    for name in TABLES:
        rows=list(csv.DictReader(io.StringIO(portable.canonical(directory/(name+'.csv')).decode('utf-8'))))
        parsed[name]=rows
        if name=='radial_profiles':continue
        expected=metrics['detection_rows'] if name=='detections' else metrics[name]
        if len(rows)!=len(expected):raise ValueError('CSV row count '+name)
        for row,target in zip(rows,expected):
            if set(row)!=set(target):raise ValueError('CSV columns '+name)
            for key,value in target.items():
                if value is None:
                    if row[key]!='':raise ValueError('CSV null '+key)
                elif isinstance(value,(bool,str)):
                    if row[key]!=str(value):raise ValueError('CSV label '+key)
                else:close(float(row[key]),value,'CSV '+key)
    if len(metrics['measurements'])!=60 or len(metrics['parameters'])!=13 or len(metrics['boundaries'])!=3 or len(parsed['radial_profiles'])!=1503:
        raise ValueError('Scientific row counts differ.')
    with np.load(directory/'reference_arrays.npz',allow_pickle=False) as z:arrays={n:z[n] for n in z.files}
    scenes=reference.truth();shape=(96,128);clean=scenes['vertical']['clean'];expected_arrays={'clean':clean}
    if set(metrics['scenes'])!=set(scenes):raise ValueError('Scene inventory mismatch')
    all_responses={}
    for name,scene in scenes.items():
        spec={k:scene[k] for k in ['components','gaussian_sigma','seed']}
        if metrics['scenes'][name]!=spec:raise ValueError('Scene parameters differ: '+name)
        hset=reference.responses(shape,scene['components']);all_responses[name]=hset
        for field in ['periodic','gaussian','noisy']:
            key = 'gaussian_noise' if field=='gaussian' else field
            expected_arrays[name+'_'+key]=scene[field]
        for method,h in hset.items():
            output=np.fft.ifft2(np.fft.fft2(scene['noisy'])*h).real
            if method!='noisy':expected_arrays[name+'_'+method]=output
            rows=[r for r in metrics['measurements'] if r['scene']==name and r['method']==method]
            if len(rows)!=1:raise ValueError('Missing/duplicate measurement')
            for k,v in reference.quality(scene,h).items():close(rows[0][k],v,name+'/'+method+'/'+k)
    scene=scenes['vertical'];centers=[c['frequency_cpp'] for c in scene['components']]
    for method in ['ideal','butterworth','gaussian','radial_butterworth','lowpass']:expected_arrays['response_'+method]=all_responses['vertical'][method]
    expected_arrays['vertical_spectrum']=np.fft.fft2(scene['noisy'])
    expected_settings=[('radius',k,r,2) for r in [.004,.012,.035] for k in ['ideal','butterworth','gaussian']]+[('order','butterworth',.012,n) for n in [1,2,4,8]]
    for row,(sweep,kind,r,n) in zip(metrics['parameters'],expected_settings):
        close(row['sweep'],sweep,'sweep');close(row['kind'],kind,'kind');close(row['radius'],r,'radius');close(row['order'],n,'order')
        h=reference.notch(shape,centers,r,kind,n)
        for k,v in (reference.quality(scene,h)|reference.ringing(h)).items():close(row[k],v,'parameters/'+k)
    index=0
    for kind in ['ideal','butterworth','gaussian']:
        for d in np.linspace(0,.5,501):
            row=parsed['radial_profiles'][index];index+=1
            if row['kind']!=kind:raise ValueError('Radial label')
            close(float(row['distance_cpp']),d,'radial distance')
            if kind=='ideal':b=float(abs(d-.1875)>.012);n=float(d>.012)
            elif kind=='butterworth':
                delta=abs(d*d-.1875**2);b=delta**4/(delta**4+(d*.024)**4);n=d**4/(d**4+.012**4)
            else:
                b=1. if d==0 else 1-np.exp(-((d*d-.1875**2)/(d*.024))**2)
                n=1-np.exp(-d*d/(2*.012**2))
            close(float(row['band_gain']),b,'band profile');close(float(row['single_notch_gain']),n,'notch profile')
    expected_proposals={n:reference.proposals(s['noisy']) for n,s in scenes.items()}
    expected_proposals['clean_only']=reference.proposals(clean)
    if set(metrics['detections'])!=set(expected_proposals):raise ValueError('Detection sets differ')
    expected_detection_rows=[]
    for name,proposals in expected_proposals.items():
        actual=metrics['detections'][name]
        if len(actual)!=len(proposals):raise ValueError('Detection proposal count')
        for rank,(a,p) in enumerate(zip(actual,proposals),start=1):
            for key,value in p.items():close(a[key],value,'detector/'+key)
            if a['interpretation']!='candidate spectral structure; not a noise classification':raise ValueError('Detector interpretation changed')
            truth=[] if name=='clean_only' else scenes[name]['components'];distances=[]
            for component in truth:
                for sign in [-1,1]:
                    delta=np.abs(np.array(p['frequency_cpp'])-sign*np.array(component['frequency_cpp']));delta=np.minimum(delta,1-delta);distances.append(float(np.linalg.norm(delta)))
            expected_detection_rows.append(dict(scene=name,rank=rank,row=p['index'][0],column=p['index'][1],fy_cpp=p['frequency_cpp'][0],fx_cpp=p['frequency_cpp'][1],partner_row=p['partner_index'][0],partner_column=p['partner_index'][1],magnitude=p['magnitude'],contrast_score=p['contrast_score'],matched_known_noise=bool(distances and min(distances)<.02)))
    if len(metrics['detection_rows'])!=len(expected_detection_rows):raise ValueError('Detection table size')
    for a,b in zip(metrics['detection_rows'],expected_detection_rows):
        for k,v in b.items():close(a[k],v,'detection table/'+k)
    crop=np.s_[10:83,9:110];bclean=clean[crop];bnoisy=scene['noisy'][crop]
    expected_arrays.update(boundary_clean=bclean,boundary_noisy=bnoisy)
    for row,(name,pad,mode) in zip(metrics['boundaries'],[('periodic',0,'reflect'),('reflect',24,'reflect'),('zero',24,'constant')]):
        padded=np.pad(bnoisy,((pad,pad),(pad,pad)),mode=mode)
        h=reference.notch(padded.shape,centers,.012,'butterworth')
        full=np.fft.ifft2(np.fft.fft2(padded)*h);out=full.real[pad:pad+73,pad:pad+101]
        expected_arrays['boundary_'+name]=out
        expected=dict(mode=name,padding_rows=pad,padding_columns=pad,padded_rows=padded.shape[0],padded_columns=padded.shape[1],mse=float(np.mean((out-bclean)**2)),psnr_db=float(-10*np.log10(np.mean((out-bclean)**2))),imaginary_residual=float(np.max(abs(full.imag))))
        for k,v in expected.items():close(row[k],v,'boundary/'+k)
    for key,source,locations in [
        ('damaged_legitimate',clean,[(18/96,0)]),
        ('automatic_clean',clean,[p['frequency_cpp'] for p in expected_proposals['clean_only']]),
        ('automatic_noisy',scene['noisy'],[p['frequency_cpp'] for p in expected_proposals['vertical']])]:
        h=reference.notch(shape,locations,.012,'butterworth');expected_arrays[key]=np.fft.ifft2(np.fft.fft2(source)*h).real
    failure=dict(legitimate_notch_mse=float(np.mean((expected_arrays['damaged_legitimate']-clean)**2)),automatic_clean_mse=float(np.mean((expected_arrays['automatic_clean']-clean)**2)),automatic_noisy_mse=float(np.mean((expected_arrays['automatic_noisy']-clean)**2)))
    for k,v in failure.items():close(metrics['failure'][k],v,'failure metric')
    step=np.tile((np.arange(128)>=64).astype(float),(96,1));expected_arrays['step']=step
    for kind in ['ideal','butterworth','gaussian']:expected_arrays['step_'+kind]=np.fft.ifft2(np.fft.fft2(step)*all_responses['vertical'][kind]).real
    if set(arrays)!=set(expected_arrays) or set(metrics['array_inventory'])!=set(expected_arrays) or len(arrays)!=99:
        raise ValueError('Reference array inventory mismatch')
    for name,expected in expected_arrays.items():
        a=arrays[name];dtype='complex128' if name=='vertical_spectrum' else 'float64'
        if a.shape!=expected.shape or a.dtype!=dtype or not np.isfinite(a).all():raise ValueError('Array shape/dtype/finiteness '+name)
        if metrics['array_inventory'][name]!=dict(shape=list(expected.shape),dtype=dtype):raise ValueError('Array metadata mismatch')
        close(a,expected,'array/'+name)
    if metrics['clean_reference']!=dict(mse=0.,psnr_db=None,psnr_infinite=True):raise ValueError('Clean quality reference mismatch')
    return dict(figures=12,arrays=len(arrays),table_rows={k:len(v) for k,v in parsed.items()},files=file_reports,
                independent_scientific_checks='passed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',nargs='?',type=Path,default=Path(__file__).resolve().parent/'output')
    print(json.dumps(audit(p.parse_args().directory),indent=2))
