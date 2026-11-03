"""Scientific auditor: independent full maps, metadata, metrics and binary integrity."""
import argparse,csv,io,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from mean_support import portable
import mean_audit_reference as ref
from mean_reference import decimal_mean,coordinate

FIGURES='overview arithmetic_geometric harmonic q_sweep wrong_q mixed window_sizes profiles boundaries noise_sweep stress quantitative'.split()
TABLES=['main','q_sweep','windows','boundaries','noise_sweep','stress']
EXPECTED={n+'.png' for n in FIGURES}|{n+'.csv' for n in TABLES}|{'metrics.json','reference_arrays.npz'}
METHODS=['arithmetic','geometric','harmonic','contra']
QS=[-1.5,-1.,-.5,0.,.5,1.,1.5]
WINDOWS=[(1,1),(1,3),(3,1),(3,3),(5,5),(7,7),(3,5)]
DETAILS=dict(dark_line=dict(row=36,columns=[6,32],profile_column=20,profile_rows=[31,42]),
             bright_line=dict(row=44,columns=[46,74],profile_column=60,profile_rows=[39,50]),
             bright_point=[28,15],dark_point=[28,25],step_column=40)


def close(actual,expected,label,*,rtol=1e-10,atol=1e-12):
    if expected is None:
        if actual is not None:raise ValueError('Expected null: '+label)
    elif isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('Field mismatch: '+label)
        for k in expected:close(actual[k],expected[k],label+'.'+k,rtol=rtol,atol=atol)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError('List mismatch: '+label)
        for k,(a,b) in enumerate(zip(actual,expected)):close(a,b,label+f'[{k}]',rtol=rtol,atol=atol)
    elif isinstance(expected,(str,bool)):
        if actual!=expected:raise ValueError('Label mismatch: '+label)
    elif actual is None:raise ValueError('Unexpected null: '+label)
    elif not np.allclose(actual,expected,rtol=rtol,atol=atol):raise ValueError('Scientific mismatch: '+label)


def audit(directory):
    directory=Path(directory)
    def read(path):
        def bad(x):raise ValueError('Nonfinite JSON '+x)
        return json.loads(portable.canonical(path),parse_constant=bad)
    entries=read(directory/'artifact_manifest.json')['files']
    if len(entries)!=len(EXPECTED) or {r['path'] for r in entries}!=EXPECTED:raise ValueError('Manifest inventory mismatch')
    if {p.name for p in directory.iterdir() if p.is_file()}!=EXPECTED|{'artifact_manifest.json'}:raise ValueError('Output inventory mismatch')
    crlf=[]
    for row in entries:
        path=directory/row['path'];data=portable.canonical(path)
        policy='utf8-crlf-to-lf' if path.suffix in ('.csv','.json') else 'raw-binary'
        if len(data)!=row['canonical_bytes'] or portable.digest(data)!=row['sha256'] or row['policy']!=policy:raise ValueError('Integrity failure: '+path.name)
        if data!=path.read_bytes():crlf.append(path.name)
        if path.suffix=='.png':
            with Image.open(path) as image:
                if image.format!='PNG':raise ValueError('Expected PNG')
                image.verify()
            with Image.open(path) as image:image.load()
    with np.load(directory/'reference_arrays.npz',allow_pickle=False) as z:a={key:z[key] for key in z.files}
    if any(v.dtype.kind not in 'fibu' or not np.isfinite(v).all() for v in a.values()):raise ValueError('Array type/finiteness failure')
    m=read(directory/'metrics.json')
    settings=dict(shape=[64,80],default_window=[3,3],default_mode='reflect',default_Q=1.5,data_range=1.,line_threshold_resolution=1e-12,
                  q_values=QS,windows=[list(w) for w in WINDOWS],sweep_seeds=[601,602,603],sweep_sigmas=[0.,.01,.03,.06],details=DETAILS,
                  zero_policy='geometric/harmonic/negative-Q: any zero -> continuous limit zero; positive-Q all-zero -> zero')
    if m['settings']!=settings or set(m)!={'settings','scenes',*TABLES}:raise ValueError('Settings mismatch')
    scenes=ref.scene_truth();positive=ref.image_truth();step=np.broadcast_to(np.where(np.arange(80)<40,.25,.75),(64,80)).copy()
    truth={};source_keys=set();stress_keys=set();targets={t:[] for t in TABLES}
    def put(key,value,source=False,stress=False):
        truth[key]=np.asarray(value)
        if source:source_keys.add(key)
        if stress:stress_keys.add(key)
    put('positive_clean',positive,True);put('black_clean',scenes['black_salt']['clean'],True);put('step_clean',step,True);put('calibration_clean',np.full((32,40),.55),True)
    scene_metadata={}
    def row(name,method,result,key,window=(3,3),Q=1.5,mode='reflect',cval=0.):
        scene=scenes[name]
        return dict(scene=name,method=method,Q=Q if method=='contra' else None,window_h=window[0],window_w=window[1],mode=mode,cval=cval,output_key=key,
                    **ref.quality_ref(scene['clean'],scene['observed'],result,scene['mask']),**ref.detail_ref(scene['clean'],result))
    for name,scene in scenes.items():
        for field in ['observed','noise','mask','labels']:put(name+'_'+field,scene[field],True)
        scene_metadata[name]=dict(seed=scene['seed'],parameters=scene['parameters'],clean_key='black_clean' if name.startswith('black_') else 'positive_clean',corruption_fraction=float(scene['mask'].mean()),minimum_observed=float(scene['observed'].min()))
        for method in METHODS:
            key=name+'_'+method;result=ref.full_filter(scene['observed'],method);put(key,result)
            targets['main'].append(row(name,method,result,key))
    for name,meta in scene_metadata.items():
        for field in ['seed','parameters','clean_key']:
            if m['scenes'][name][field]!=meta[field]:raise ValueError('Scene parameter mismatch')
    close(m['scenes'],scene_metadata,'scene metadata')
    for name in ['salt','pepper','mixed','black_salt','black_pepper','black_mixed']:
        for index,Q in enumerate(QS):
            key=f'q_{name}_{index}';result=ref.full_filter(scenes[name]['observed'],'contra',Q=Q);put(key,result)
            targets['q_sweep'].append(row(name,'contra',result,key,Q=Q))
    for index,window in enumerate(WINDOWS):
        for method in METHODS:
            key=f'window_{index}_{method}';result=ref.full_filter(scenes['gaussian']['observed'],method,window);put(key,result)
            clean_result=ref.full_filter(positive,method,window);edge_result=ref.full_filter(step,method,window)
            put(key+'_clean',clean_result);put(key+'_edge',edge_result)
            entry=row('gaussian',method,result,key,window)
            entry.update(clean_output_key=key+'_clean',edge_output_key=key+'_edge',clean_mse=float(np.mean((clean_result-positive)**2)),**ref.edge_ref(edge_result))
            targets['windows'].append(entry)
    border=np.ones((64,80),bool);border[3:-3,3:-3]=False;put('border_mask',border,True)
    for mode in ['reflect','edge','constant']:
        for label,method,Q in [('arithmetic','arithmetic',1.5),('geometric','geometric',1.5),('harmonic','harmonic',1.5),('contra','contra',1.5),('contra_negative','contra',-1.5)]:
            key=f'boundary_{mode}_{label}';result=ref.full_filter(scenes['gaussian']['observed'],method,(7,7),Q=Q,mode=mode);put(key,result)
            entry=row('gaussian',method,result,key,(7,7),Q,mode)
            delta=result-positive
            entry.update(label=label,border_mse=float(np.mean(delta[border]**2)),interior_mse=float(np.mean(delta[~border]**2)),border_bias=float(np.mean(delta[border])))
            targets['boundaries'].append(entry)
    calibration=np.full((32,40),.55)
    for level,sigma in enumerate([0.,.01,.03,.06]):
        for seed in [601,602,603]:
            eta=np.random.default_rng(seed).normal(0.,sigma,(32,40));observed=calibration+eta;prefix=f'sweep_{level}_{seed}'
            put(prefix+'_noise',eta,True);put(prefix+'_observed',observed,True)
            for method in METHODS:
                key=prefix+'_'+method;result=ref.full_filter(observed,method);put(key,result)
                delta=(result-calibration)[1:-1,1:-1]
                targets['noise_sweep'].append(dict(sigma=sigma,seed=seed,method=method,output_key=key,**ref.quality_ref(calibration,observed,result,eta!=0),interior_residual_variance=float(np.var(delta)),theory_arithmetic_variance=sigma*sigma/9))
    stress=[('tiny',[1e-300,2e-300,4e-300],1.5),('huge',[1e300,2e300,4e300],1.5),('span',[1e-300,1.,1e300],-.5),('zero',[0.,.5,1.],-1.5),('allzero',[0.,0.,0.],1.5),('near_huge',[1e200*(1-2e-12),1e200*(1-1e-12),1e200],1e12),('positive_limit',[1.,2.,4.],np.finfo(float).max),('negative_limit',[1.,2.,4.],-np.finfo(float).max),('near_max',[np.nextafter(np.finfo(float).max,0),np.finfo(float).max,np.finfo(float).max],-.5)]
    for case,values,Q in stress:
        source=np.array([values]);put('stress_'+case+'_input',source,True,True)
        for method in METHODS:
            key='stress_'+case+'_'+method;result=np.zeros(source.shape);limit=method=='contra' and abs(Q)>1e100
            for y,x in np.ndindex(source.shape):
                local=[float(source[0,coordinate(x+j-len(values)//2,len(values),'reflect')]) for j in range(len(values))]
                result[y,x]=float(max(local) if Q>0 else min(local)) if limit else decimal_mean(local,method,Q)
            put(key,result,stress=True)
            expected=float(result[0,len(values)//2]);actual=float(a[key][0,len(values)//2])
            # Reported roundoff errors must match stored arrays, not a fabricated 0.
            targets['stress'].append(dict(case=case,method=method,Q=Q if method=='contra' else None,output_key=key,actual=actual,reference=expected,reference_kind='concentration_limit' if limit else 'decimal100',absolute_error=abs(actual-expected),relative_error=abs(actual-expected)/abs(expected) if expected else 0.,ulp_error=abs(actual-expected)/math.ulp(expected)))
    if set(a)!=set(truth):raise ValueError('NPZ key inventory mismatch')
    max_error=0.;max_relative=0.;ordinary_error=0.
    for key,expected in truth.items():
        actual=a[key]
        if actual.shape!=expected.shape or actual.dtype!=expected.dtype:raise ValueError('Array shape/dtype mismatch: '+key)
        if expected.dtype.kind in 'biu':
            if not np.array_equal(actual,expected):raise ValueError('Discrete array mismatch: '+key)
        elif key in source_keys:
            atol=0. if key in stress_keys else 16*np.finfo(float).eps*float(np.max(abs(expected)))
            close(actual,expected,'source '+key,rtol=4*np.finfo(float).eps,atol=atol)
            if not np.array_equal(actual[expected==0],expected[expected==0]):raise ValueError('Source exact-zero mismatch')
        else:
            close(actual,expected,'map '+key,atol=0. if key in stress_keys else 1e-12)
            if not np.array_equal(actual[expected==0],expected[expected==0]):raise ValueError('Exact zero convention mismatch: '+key)
            delta=np.max(abs(actual-expected));max_error=max(max_error,float(delta))
            if key not in stress_keys:ordinary_error=max(ordinary_error,float(delta))
            nonzero=expected!=0
            if nonzero.any():max_relative=max(max_relative,float(np.max(abs((actual[nonzero]-expected[nonzero])/expected[nonzero]))))
    for table,rows in targets.items():
        close(m[table],rows,'table '+table)
        for actual,expected in zip(m[table],rows):
            for key in ['scene','method','Q','window_h','window_w','mode','cval','output_key','seed','sigma','case','reference_kind']:
                if key in expected and actual[key]!=expected[key]:raise ValueError('Scientific parameter mismatch')
    # Independently enforce special Q identities on the saved full maps.
    for name in ['salt','pepper','mixed','black_salt','black_pepper','black_mixed']:
        close(a[f'q_{name}_3'],a[name+'_arithmetic'],'Q=0 identity')
        close(a[f'q_{name}_1'],a[name+'_harmonic'],'Q=-1 identity')
    for table in TABLES:
        parsed=list(csv.DictReader(io.StringIO(portable.canonical(directory/(table+'.csv')).decode('utf-8'))))
        if len(parsed)!=len(m[table]):raise ValueError('CSV row count')
        for saved,rowdata in zip(parsed,m[table]):
            if set(saved)!=set(rowdata):raise ValueError('CSV columns')
            for key,value in rowdata.items():
                if value is None:
                    if saved[key]!='':raise ValueError('CSV null')
                elif isinstance(value,str):close(saved[key],value,'CSV label')
                elif float(saved[key])!=float(value):raise ValueError('CSV/JSON value mismatch')
    return dict(figures=len(FIGURES),arrays=len(a),tables={t:len(targets[t]) for t in TABLES},full_independent_maps=True,
                max_absolute_map_error=max_error,max_relative_map_error=max_relative,ordinary_max_absolute_map_error=ordinary_error,normalized_crlf=crlf,scientific_checks='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',nargs='?',type=Path,default=Path(__file__).resolve().parent/'output');print(json.dumps(audit(parser.parse_args().directory),indent=2))
