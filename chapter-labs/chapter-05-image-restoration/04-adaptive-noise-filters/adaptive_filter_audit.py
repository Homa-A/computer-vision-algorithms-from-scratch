"""Canonical integrity plus independent exact-formula, Stage A/B and metric audit."""
import argparse,csv,io,json,math
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from adaptive_design import design,masks,stress_inputs,ROI,BAD_ROI,FIGURES,GROUPS
from adaptive_reference import reference,neighborhood
from adaptive_support import portable,figure_io


def require(ok,message):
    if not ok:raise ValueError(message)


def close(a,b,message,rtol=1e-11,atol=1e-13):
    if b is None:require(a is None,message)
    elif b==0:require(a==0,message)
    else:require(a is not None and math.isfinite(a) and math.isclose(a,b,rel_tol=rtol,abs_tol=atol),message)


def avg(v):
    v=np.asarray(v).ravel();return math.fsum(map(float,v))/len(v) if len(v) else None


def variance(v):
    z=[F.from_float(float(x)) for x in np.asarray(v).ravel()];m=sum(z,F())/len(z)
    return float(sum(((x-m)**2 for x in z),F())/len(z))


def observed(c):
    """Direct independent draw formulas, no generator/noise API calls."""
    a=c['clean'].copy();kind=c['kind'];rng=np.random.default_rng(c['seed']);labels=np.zeros(a.shape,np.int8)
    if kind in ('gaussian','nonuniform'):
        delta=c['sigma']*rng.standard_normal(a.shape)
        if kind=='nonuniform':delta[:,:a.shape[1]//2]=0.
        a+=delta
    elif kind in ('salt','pepper','mixed'):
        p=c['rate'];pp=p if kind=='pepper' else 0. if kind=='salt' else p/2;ps=p if kind=='salt' else 0. if kind=='pepper' else p/2
        u=rng.random(a.shape);labels[u<pp]=-1;labels[(u>=pp)&(u<pp+ps)]=1;a[labels==-1]=0.;a[labels==1]=1.
    elif kind=='border':a[0,[0,-1]]=1.;labels[0,[0,-1]]=1;a[-1,[0,-1]]=0.;labels[-1,[0,-1]]=-1
    events=labels!=0
    if kind=='gaussian':events=np.ones(a.shape,bool)
    if kind=='nonuniform':events[:,a.shape[1]//2:]=True
    return a,labels,events


def metrics(clean,noisy,out,events):
    e=out-clean;sq=e*e;mse=avg(sq);bias=avg(e);b=np.zeros(clean.shape,bool)
    for y,x in np.ndindex(clean.shape):b[y,x]=y<2 or x<2 or y>=len(clean)-2 or x>=clean.shape[1]-2
    grad=[((out[y,x+1]-out[y,x])-(clean[y,x+1]-clean[y,x]))**2 for y in range(len(clean)) for x in range(clean.shape[1]-1)]
    return dict(mse=mse,rmse=math.sqrt(mse),mae=avg(abs(e)),psnr_db=-10*math.log10(mse) if mse else None,bias=bias,
        variance=0. if np.ptp(e)==0 else avg((e-bias)**2),corrupted_mse=avg(sq[events]),uncorrupted_mse=avg(sq[~events]),
        correction_rate=avg((abs(e[events])<=.05).astype(float)),false_alteration_rate=avg((abs(out[~events]-noisy[~events])>1e-12).astype(float)),
        gradient_rmse=math.sqrt(avg(grad)),border_mse=avg(sq[b]),interior_mse=avg(sq[~b]))


def check_maps(a,key,params,row,shape,keys):
    method=params['method']
    fields=('mean','variance','weight') if method=='local' else ('size','expansions','stage_a_passed','stage_b_replaced','retained','fallback','stage_a_failures','minimum','median','maximum') if method=='adaptive_median' else ()
    d={}
    for field in fields:
        k=key+'__'+field;keys.add(k);v=a[k];d[field]=v
        dtype=np.int64 if field in ('size','expansions','stage_a_failures') else np.bool_ if field in ('stage_a_passed','stage_b_replaced','retained','fallback') else np.float64
        require(v.shape==shape and v.dtype==dtype,'Diagnostic shape/dtype')
    if method=='adaptive_median':
        require(np.array_equal(d['expansions'],(d['size']-params['initial_size'])//2),'Expansion maps')
        require(np.all((d['size']>=params['initial_size'])&(d['size']<=params['max_size'])&((d['size']-params['initial_size'])%2==0)),'Selected sizes')
        require(np.array_equal(d['fallback'],~d['stage_a_passed']) and np.array_equal(d['retained'],~d['stage_b_replaced']),'Decision complement')
        require(not np.any(d['fallback']&d['stage_b_replaced']),'Fallback cannot replace')
        require(np.array_equal(d['stage_a_failures'],d['expansions']+d['fallback'].astype(np.int64)),'A failure maps')
        counts=dict(window_histogram=dict(sorted(Counter(map(str,map(int,d['size'].ravel()))).items())),expansions=sum(map(int,d['expansions'].ravel())),
            stage_a_failures=sum(map(int,d['stage_a_failures'].ravel())),stage_a_failed_pixels=int(np.count_nonzero(d['fallback'])),stage_b_replacements=int(np.count_nonzero(d['stage_b_replaced'])),retained=int(np.count_nonzero(d['retained'])),fallbacks=int(np.count_nonzero(d['fallback'])))
        if row['group']!='stress':
            for field,v in counts.items():require(row[field]==v,'Diagnostic count/histogram')
    return d


def audit(directory):
    directory=Path(directory);expected={n+'.png' for n in FIGURES}|{g+'.csv' for g in GROUPS}|{'metrics.json','reference_arrays.npz','artifact_manifest.json'}
    require({p.name for p in directory.iterdir() if p.is_file()}==expected,'File inventory')
    manifest=json.loads(portable.canonical(directory/'artifact_manifest.json'))
    require(len(manifest['files'])==len(expected)-1 and {r['path'] for r in manifest['files']}==expected-{'artifact_manifest.json'},'Manifest inventory')
    for r in manifest['files']:
        b=portable.canonical(directory/r['path']);require(len(b)==r['canonical_bytes'] and portable.digest(b)==r['sha256'],'Artifact hash/size')
        require(r['policy']==('utf8-crlf-to-lf' if Path(r['path']).suffix in ('.csv','.json') else 'raw-binary'),'Integrity policy')
    for n in FIGURES:figure_io.verify_png(directory/(n+'.png'))
    p=json.loads(portable.canonical(directory/'metrics.json'));cases,tasks=design();byid={c['id']:c for c in cases};mm=masks()
    require(p['schema']==1 and p['range']==1. and p['roi']==list(ROI) and p['contaminated_roi']==list(BAD_ROI) and p['correction_tolerance']==.05 and p['alteration_tolerance']==1e-12 and p['structure_tolerance']==.1,'Conventions')
    require(p['cases']==[{k:v for k,v in c.items() if k!='clean'} for c in cases] and p['figures']==list(FIGURES) and p['groups']==list(GROUPS),'Design')
    with np.load(directory/'reference_arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    require(len(a)==p['array_count'] and all(v.dtype.kind in 'fib' and np.isfinite(v).all() for v in a.values()),'Array inventory/finiteness')
    keys=set()
    for n,m in mm.items():
        k='mask_'+n;keys.add(k);require(a[k].dtype==np.bool_ and np.array_equal(a[k],m),'Structure mask')
    for c in cases:
        n=c['id'];noisy,labels,events=observed(c)
        for s,v in dict(clean=c['clean'],noisy=noisy,noise=noisy-c['clean'],labels=labels,events=events).items():
            k=n+'_'+s;keys.add(k);require(a[k].dtype==v.dtype and a[k].shape==v.shape,'Source shape/dtype')
            require(np.array_equal(a[k],v) if v.dtype.kind in 'bi' else np.allclose(a[k],v,rtol=8*np.finfo(float).eps,atol=0),'Source/noise ground truth')
    rows=p['rows'];require(len(rows)==len(tasks)+12,'Row count');count=0;max_abs=max_rel=0.;max_var_rel=0.
    for i,(n,g,original) in enumerate(tasks):
        row=rows[i];key=f'result_{i:04d}';keys.add(key);params=row['params'];c=byid[n]
        require(row['id']==key and row['case']==n and row['group']==g,'Row identity')
        for k,v in original.items():require(params[k]==v,'Filter parameters')
        clean,noisy,events=(a[n+'_'+s] for s in ('clean','noisy','events'));out=a[key]
        require(out.shape==clean.shape and out.dtype==np.float64,'Result shape/dtype')
        if params['method']=='local':
            source=params['variance_source'];roi=BAD_ROI if source=='contaminated' else ROI
            base=variance(noisy[roi[0]:roi[1],roi[2]:roi[3]]) if source in ('estimated','contaminated') else .05**2 if source=='assumed_mismatch' else c['sigma']**2
            close(params['noise_variance'],base*params['factor'],'Independent observation-only noise estimate')
            truth=c['sigma']**2;close(row['variance_error'],params['noise_variance']-truth,'Variance error')
            close(row['variance_relative_error'],params['noise_variance']/truth-1 if truth else None,'Variance relative error')
        d=check_maps(a,key,params,row,out.shape,keys)
        h,w=clean.shape;points=list(np.ndindex(clean.shape)) if n=='border' else sorted({(0,0),(0,w-1),(h-1,0),(h-1,w-1),(h//2,w//2),(min(14,h-1),9),(min(20,h-1),12),(h//3,w//3),(h//2-1,w//2-1)})
        if params['method'] in ('local','adaptive_median'):
            expected,diags=reference(noisy,params['method'],params,points)
            for field,v in d.items():
                actual=np.array([v[y,x] for y,x in points]);ref=np.array([r[field] for r in diags])
                require(np.array_equal(actual,ref) if v.dtype.kind in 'bi' else np.allclose(actual,ref,rtol=1e-10,atol=1e-14),'Independent diagnostic '+field)
                if field=='variance':max_var_rel=max(max_var_rel,float((abs(actual-ref)/np.maximum(abs(ref),np.finfo(float).tiny)).max()))
        else:
            expected=[]
            for y,x in points:
                z=sorted(neighborhood(noisy,y,x,params['window'],params['mode'],params['cval']))
                expected.append(z[len(z)//2] if params['method']=='fixed_median' else float(sum(map(F.from_float,z),F())/len(z)))
            expected=np.array(expected)
        selected=np.array([out[y,x] for y,x in points]);require(np.allclose(selected,expected,rtol=1e-10,atol=1e-13),'Independent output reference')
        if params['method'] in ('adaptive_median','fixed_median'):require(np.array_equal(selected,expected),'Exact median output')
        diff=abs(selected-expected);max_abs=max(max_abs,float(diff.max()));max_rel=max(max_rel,float((diff/np.maximum(abs(expected),np.finfo(float).tiny)).max()));count+=len(points)
        for k,v in metrics(clean,noisy,out,events).items():close(row[k],v,'Metric '+k)
        if clean.shape==(32,44):
            for name,mask in mm.items():
                close(row[name+'_mse'],avg((out[mask]-clean[mask])**2),'Region error')
                close(row[name+'_survival'],avg((abs(out[mask]-clean[mask])<=.1).astype(float)),'Structure survival')
        if params['method']=='adaptive_median':
            require(np.array_equal(out[d['retained']],noisy[d['retained']]),'Original-center retention/fallback')
            require(np.array_equal(out[d['stage_b_replaced']],d['median'][d['stage_b_replaced']]),'Median replacement')
    # Full stress maps against exact rational / recursive independent oracles.
    stress_abs=stress_rel=0.
    require([(r['case'],r['params']['method']) for r in rows[len(tasks):]]==[(n,m) for n in stress_inputs() for m in ('local','adaptive_median')],'Stress row inventory')
    for row in rows[len(tasks):]:
        key=row['id'];source='stress_'+row['case'];keys.update((source,key,key+'__reference'))
        image=a[source];require(image.dtype==np.float64 and np.array_equal(image,stress_inputs()[row['case']]),'Stress source');params=row['params'];out=a[key]
        expected_params=dict(method='local',window=[1,3],regularization='clipped',variance_source='oracle',factor=1.,noise_variance=.0025,mode='reflect',cval=0.) if params['method']=='local' else dict(method='adaptive_median',initial_size=3,max_size=5,mode='reflect',cval=0.)
        require(params==expected_params and row['group']=='stress' and key==source+'_'+params['method'],'Stress parameters')
        expected,diags=reference(image,params['method'],params);expected=expected.reshape(image.shape)
        require(out.shape==image.shape and out.dtype==np.float64 and a[key+'__reference'].dtype==np.float64 and np.array_equal(a[key+'__reference'],expected),'Stored stress oracle')
        require(np.allclose(out,expected,rtol=1e-12,atol=0),'Stress output reference')
        d=check_maps(a,key,params,row,out.shape,keys)
        for field,v in d.items():
            ref=np.array([r[field] for r in diags]).reshape(image.shape)
            require(np.array_equal(v,ref) if v.dtype.kind in 'bi' else np.allclose(v,ref,rtol=1e-12,atol=0),'Stress diagnostic '+field)
        diff=abs(out-expected);rel=float((diff/np.maximum(abs(expected),np.finfo(float).tiny)).max())
        close(row['max_absolute_error'],float(diff.max()),'Stress error',atol=0);close(row['max_relative_error'],rel,'Stress relative',atol=0)
        stress_abs=max(stress_abs,float(diff.max()));stress_rel=max(stress_rel,rel)
    require(set(a)==keys,'Unexpected/missing array keys')
    for g in GROUPS:
        table=list(csv.DictReader(io.StringIO(portable.canonical(directory/(g+'.csv')).decode())));rr=[r for r in rows if r['group']==g];fields=set().union(*(r.keys() for r in rr))
        require(len(table)==len(rr) and set(table[0])==fields,'CSV schema/count')
        for entry,row in zip(table,rr):
            for k in fields:
                v=row.get(k);text=entry[k]
                if v is None:require(text=='','CSV null')
                elif isinstance(v,str):require(text==v,'CSV string')
                elif isinstance(v,(dict,list)):require(json.loads(text)==v,'CSV structured value')
                else:require(float(text)==v,'CSV numerical coherence')
    return dict(scientific_checks='passed',arrays=len(a),rows=len(rows),figures=len(FIGURES),tables=len(GROUPS),independent_pixels=count,
        ordinary_max_absolute_error=max_abs,ordinary_max_relative_error=max_rel,local_variance_max_relative_error=max_var_rel,stress_max_absolute_error=stress_abs,stress_max_relative_error=stress_rel)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',nargs='?',type=Path,default=Path(__file__).resolve().parent/'output')
    print(json.dumps(audit(parser.parse_args().directory),indent=2,allow_nan=False))
