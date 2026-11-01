"""Independent scientific map/localization audit and portable artifact integrity."""
import argparse,csv,io,json
from pathlib import Path
import numpy as np
from scipy.signal import convolve2d,correlate2d
from PIL import Image
from matching_support import portable
import matching_reference as ref

FIGURES='localization direct_fft padding_geometry short_padding ranking affine noise_sweep ambiguity occlusion_distractors transformation_limits size_comparison quantitative'.split()
TABLES=['measurements','sweeps','detection_trials','benchmark']
EXPECTED={n+'.png' for n in FIGURES}|{n+'.csv' for n in TABLES}|{'metrics.json','reference_arrays.npz'}


def close(a,b,label):
    if b is None:
        if a is not None:raise ValueError('Undefined value mismatch: '+label)
    elif isinstance(b,(str,bool)):
        if a!=b:raise ValueError('Label mismatch: '+label)
    elif not np.allclose(a,b,rtol=1e-10,atol=1e-10):raise ValueError('Scientific mismatch: '+label)


def quality(name,method,values,valid,locations,template_shape):
    info=ref.localization(values,valid,(template_shape[0]//2,template_shape[1]//2));loc=info['location']
    error=None if not locations or loc is None else float(min(np.hypot(loc[0]-y,loc[1]-x) for y,x in locations))
    true=[float(values[y,x]) for y,x in locations]
    return dict(scene=name,method=method,row=loc[0] if loc is not None else None,column=loc[1] if loc is not None else None,score=info['score'],second_score=info['peaks'][1]['score'] if len(info['peaks'])>1 else None,
                true_score=max(true) if true else None,error_pixels=error,wrong_location=bool(error>.5) if error is not None else None,
                eligible_count=info['eligible_count'],tie_count=info['tie_count'],ambiguous=info['ambiguous'],gap=info['gap'],psr=info['psr'],psr_status=info['psr_status'],
                response_rows=values.shape[0],response_columns=values.shape[1],template_rows=template_shape[0],template_columns=template_shape[1])


def audit(directory):
    directory=Path(directory)
    def read_json(p):
        def bad(v):raise ValueError('Nonfinite JSON '+v)
        return json.loads(portable.canonical(p),parse_constant=bad)
    entries=read_json(directory/'artifact_manifest.json')['files']
    if len(entries)!=len(EXPECTED) or {e['path'] for e in entries}!=EXPECTED:raise ValueError('Artifact inventory mismatch')
    if {p.name for p in directory.iterdir() if p.is_file()}!=EXPECTED|{'artifact_manifest.json'}:raise ValueError('Unexpected output files')
    reports=[]
    for e in entries:
        p=directory/e['path'];raw=p.read_bytes();data=portable.canonical(p)
        policy='utf8-crlf-to-lf' if p.suffix in ('.json','.csv') else 'raw-binary'
        if e['canonical_bytes']!=len(data) or e['sha256']!=portable.digest(data) or e['policy']!=policy:raise ValueError('Integrity failure: '+p.name)
        reports.append(dict(path=p.name,physical_bytes=len(raw),canonical_bytes=len(data),raw_sha256=portable.digest(raw),canonical_sha256=portable.digest(data),normalized_crlf=raw!=data))
        if p.suffix=='.png':
            with Image.open(p) as im:
                if im.format!='PNG':raise ValueError('Expected PNG')
                im.verify()
            with Image.open(p) as im:im.load()
    m=read_json(directory/'metrics.json');parsed={}
    for name in TABLES:
        rows=list(csv.DictReader(io.StringIO(portable.canonical(directory/(name+'.csv')).decode('utf-8'))));parsed[name]=rows
        if name=='benchmark':continue
        targets=m[name]
        if len(rows)!=len(targets):raise ValueError('CSV row count mismatch')
        for row,target in zip(rows,targets):
            if set(row)!=set(target):raise ValueError('CSV column mismatch')
            for k,v in target.items():
                if v is None:
                    if row[k]!='':raise ValueError('CSV null mismatch')
                elif isinstance(v,(str,bool)):
                    if row[k]!=str(v):raise ValueError('CSV label mismatch')
                else:close(float(row[k]),v,'CSV '+k)
    if m['settings']!=dict(seed=808,methods=['direct','fft','zncc','ssd'],tie_atol=1e-12,tie_rtol=float(4096*np.finfo(float).eps),psr_relative_std_floor=float(np.sqrt(np.finfo(float).eps))):raise ValueError('Settings mismatch')
    with np.load(directory/'reference_arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    expected={};scenes=ref.scene_truth();computed_measurements=[];computed_sweeps=[];max_map_error=0.
    if set(m['scenes'])!=set(scenes):raise ValueError('Scene inventory mismatch')
    def numerical(row,image,template):
        residual=row['imaginary_residual'];K=template.size
        if row['method'] in ['direct','ssd']:
            if residual is not None:raise ValueError('Spatial method cannot have an inverse-FFT measurement')
        else:
            scale=max(1.,float(np.max(abs(image)))*float(np.sum(abs(template)))) if row['method']=='fft' else max(1.,2*np.sqrt(K))
            if residual is None or not np.isfinite(residual) or not 0<=residual<=1e-10*scale:raise ValueError('Imaginary residual exceeds roundoff bound')
        count=row['fallback_windows']
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=(image.shape[0]-template.shape[0]+1)*(image.shape[1]-template.shape[1]+1):raise ValueError('Invalid fallback count')
        if row['method']!='zncc' and count!=0:raise ValueError('Fallback on non-normalized method')
    for name,s in scenes.items():
        spec={k:s[k] for k in ['locations','seed','sigma','present','description']};spec.update(image_shape=list(s['image'].shape),template_shape=list(s['template'].shape))
        if m['scenes'][name]!=spec:raise ValueError('Scene parameters mismatch')
        for field in ['clean','image','template']:
            key=name+'_'+field;expected[key]=s[field]
            if key not in a or not np.allclose(a[key],s[field],rtol=4*np.finfo(float).eps,atol=4*np.finfo(float).eps):raise ValueError('Scientific input mismatch: '+key)
        maps,mask=ref.maps(s['image'],s['template'])
        for method,v in maps.items():
            key=name+'_'+method;expected[key]=v
            valid=mask if method=='zncc' else np.ones(v.shape,bool)
            row=next((r for r in m['measurements'] if r['scene']==name and r['method']==method),None)
            if row is None:raise ValueError('Missing method record')
            target=quality(name,method,v,valid,s['locations'],s['template'].shape);computed_measurements.append(target)
            for k,value in target.items():close(row[k],value,name+'/'+method+'/'+k)
            numerical(row,s['image'],s['template'])
            if key in a:max_map_error=max(max_map_error,float(np.max(abs(a[key]-v))))
        expected[name+'_valid']=mask
    base=scenes['unique'];maps,_=ref.maps(base['image']*1.7+.4,base['template']);expected['affine_positive']=maps['zncc']
    maps,_=ref.maps(-base['image']*1.7+.4,base['template']);expected['affine_negative']=maps['zncc']
    specs=[]
    for kind,values in [('noise',[0.,.04,.12,.24,.4,.7]),('occlusion',[0.,.2,.4,.6,.8]),('contrast',[-1.5,-.5,.25,1.,2.,4.])]:
        for j,value in enumerate(values):
            key=f'{kind}_{j}';image,t,loc=ref.sweep_truth(kind,value);specs.append(dict(key=key,kind=kind,value=value,location=loc))
            expected[key+'_image']=image;maps,mask=ref.maps(image,t);expected[key+'_zncc']=maps['zncc'];expected[key+'_valid']=mask
            for method,v in maps.items():
                target=dict(kind=kind,value=value,**quality(key,method,v,mask if method=='zncc' else np.ones(v.shape,bool),[loc],t.shape));computed_sweeps.append(target)
                row=next(r for r in m['sweeps'] if r['scene']==key and r['method']==method)
                for k,val in target.items():close(row[k],val,'sweep/'+k)
                numerical(row,image,t)
    if m['sweep_specs']!=specs or len(m['measurements'])!=60 or len(m['sweeps'])!=68:raise ValueError('Experiment table inventory mismatch')
    for method in ['direct','fft','zncc','ssd']:
        rate=float(np.mean([r['wrong_location'] for r in computed_measurements if r['method']==method and r['wrong_location'] is not None]))
        close(m['localization_false_match_rates'][method],rate,'false-match rate')
    trials=[]
    for j in range(20):
        image=.08+np.random.default_rng(9000+j).normal(0,.03,(48,64));t=base['template'];category='background' if j<10 else 'near_distractor'
        if j>=10:image[20:29,30:41]=t+.08*np.random.default_rng(10000+j).normal(size=t.shape)
        maps,mask=ref.maps(image,t);info=ref.localization(maps['zncc'],mask,(4,5));loc=info['location']
        trials.append(dict(seed=9000+j,category=category,score=info['score'],row=loc[0],column=loc[1],threshold=.8,accepted=bool(info['score']>=.8)))
    if len(m['detection_trials'])!=20:raise ValueError('Absent-trial count')
    for actual,target in zip(m['detection_trials'],trials):
        for k,v in target.items():close(actual[k],v,'detection/'+k)
    for category in ['background','near_distractor']:close(m['absent_false_positive_rates'][category],np.mean([r['accepted'] for r in trials if r['category']==category]),'absent false-positive rate')
    g=np.arange(1.,36.).reshape(5,7)/35;t=np.array([[.2,.9],[-.4,.1],[.7,-.2]]);full=ref.full_convolution(g,t)
    expected.update(geometry_image=g,geometry_template=t,geometry_full=full,geometry_valid=full[2:5,1:7],geometry_circular=ref.circular(g,t).astype(complex))
    for name,shape in [('short_row',(6,8)),('short_column',(7,7))]:
        folded=np.zeros(shape)
        for y,x in np.ndindex(full.shape):folded[y%shape[0],x%shape[1]]+=full[y,x]
        expected[name]=folded
    t=base['template'];wrapped=np.zeros((17,23))
    for i,j in np.ndindex(t.shape):wrapped[(14+i)%17,(19+j)%23]=t[i,j]
    v,_=ref.maps(wrapped,t);expected.update(wrapped_image=wrapped,wrapped_circular=ref.circular(wrapped,t).astype(complex),wrapped_valid=v['direct'])
    # Existing SciPy is an independent direct full/valid oracle at course size.
    y,x=np.indices((38,42));t=np.full((38,42),.12);t+=.55*((x>=42//5)&(x<=42//2)&(y>=1)&(y<37));t+=.3*((y>=19)&(y<=20)&(x>=14)&(x<41));t+=.75*((x==40)&(y==1))
    course=np.zeros((256,256));course[218:,214:]=t
    full=convolve2d(course,t[::-1,::-1],mode='full');valid=correlate2d(course,t,mode='valid')
    expected.update(course_image=course,course_template=t,course_full=full,course_valid=valid)
    for k,v in dict(image_shape=[256,256],template_shape=[38,42],minimum_fft_shape=[293,297],valid_shape=[219,215],location=[218,214]).items():close(m['course'][k],v,'course/'+k)
    if not 0<=m['course']['imaginary_residual']<=1e-10*max(1,float(np.sum(abs(t)))):raise ValueError('Course inverse residual')
    if len(a)!=187 or set(a)!=set(expected) or set(m['array_inventory'])!=set(expected):raise ValueError('Reference-array inventory mismatch')
    for name,target in expected.items():
        dtype='bool' if target.dtype==bool else 'complex128' if np.iscomplexobj(target) else 'float64'
        actual=a[name]
        if actual.shape!=target.shape or actual.dtype!=dtype or not np.isfinite(actual).all():raise ValueError('Array shape/dtype/finiteness mismatch: '+name)
        if m['array_inventory'][name]!=dict(shape=list(target.shape),dtype=dtype):raise ValueError('Array metadata mismatch')
        if dtype=='bool':
            if not np.array_equal(actual,target):raise ValueError('Validity-mask mismatch')
        else:close(actual,target,'array/'+name)
    # Times/memory are measurements, never deterministic equality assertions.
    if len(parsed['benchmark'])!=12:raise ValueError('Benchmark record count')
    for row,(H,W,h,w,method) in zip(parsed['benchmark'],[(H,W,h,w,method) for H,W,h,w in [(32,48,5,7),(64,80,9,11),(128,128,15,17)] for method in ['direct','fft','zncc','ssd']]):
        for k,v in dict(image_rows=H,image_columns=W,template_rows=h,template_columns=w,repeats=3).items():close(int(row[k]),v,'benchmark '+k)
        if row['method']!=method:raise ValueError('Benchmark label')
        seconds=float(row['median_seconds']);memory=int(row['tracemalloc_peak_bytes']);error=float(row['max_absolute_error'])
        if not np.isfinite(seconds) or seconds<=0 or memory<0 or not np.isfinite(error) or not 0<=error<=1e-10*max(1,h*w):raise ValueError('Benchmark measurement/accuracy bounds')
    return dict(figures=12,arrays=len(a),tables={k:len(v) for k,v in parsed.items()},max_map_absolute_error=max_map_error,files=reports,
                timing_policy='Validate schema and finite accuracy bounds; runtime and memory are environment-specific measurements.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',nargs='?',type=Path,default=Path(__file__).resolve().parent/'output');print(json.dumps(audit(p.parse_args().directory),indent=2))
