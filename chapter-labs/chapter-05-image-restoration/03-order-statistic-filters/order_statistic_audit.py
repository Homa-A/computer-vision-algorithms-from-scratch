"""Independent scientific/portable audit. Never calls a production filter."""
import argparse,csv,io,json,math
from pathlib import Path
import numpy as np
from order_support import portable,figure_io
from order_design import design,masks,stresses,METHODS,FIGURES,GROUPS
from order_reference import reference,samples


def require(ok,message):
    if not ok:raise ValueError(message)


def close(a,b,message,rtol=1e-11,atol=1e-13):
    if b is None:require(a is None,message)
    elif b==0:require(a==0,message)
    else:require(a is not None and math.isfinite(a) and math.isclose(a,b,rel_tol=rtol,abs_tol=atol),message)


def avg(v):
    v=np.asarray(v).ravel();return math.fsum(map(float,v))/len(v) if len(v) else None


def independent_metrics(clean,noisy,out,events):
    e=out-clean;sq=e*e;mse=avg(sq);bias=avg(e)
    b=np.zeros(clean.shape,bool)
    for y,x in np.ndindex(clean.shape):b[y,x]=y<2 or x<2 or y>=len(clean)-2 or x>=clean.shape[1]-2
    gradients=[((out[y,x+1]-out[y,x])-(clean[y,x+1]-clean[y,x]))**2 for y in range(len(clean)) for x in range(clean.shape[1]-1)]
    return dict(mse=mse,rmse=math.sqrt(mse),mae=avg(abs(e)),psnr_db=-10*math.log10(mse) if mse else None,
        bias=bias,variance=avg((e-bias)**2),corrupted_mse=avg(sq[events]),uncorrupted_mse=avg(sq[~events]),
        correction_rate=avg((abs(e[events])<=.05).astype(float)),false_alteration_rate=avg((abs(out[~events]-noisy[~events])>1e-12).astype(float)),
        border_mse=avg(sq[b]),interior_mse=avg(sq[~b]),gradient_rmse=math.sqrt(avg(gradients)))


def independent_observation(c):
    """Recompute draws from explicit formulas, not generator or noise API."""
    a=c['clean'].copy();labels=np.zeros(a.shape,np.int8);k=c['kind'];sd=c['sigma'];rng=np.random.default_rng(c['seed'])
    if k=='border':
        a[0,[0,-1]]=1.;labels[0,[0,-1]]=1;a[-1,[0,-1]]=0.;labels[-1,[0,-1]]=-1
    else:
        if k in ('gaussian','gaussian_impulse'):a+=sd*rng.standard_normal(a.shape)
        if k in ('uniform','uniform_impulse') and sd>0:
            u=rng.random(a.shape);b=np.sqrt(3)*sd;a+=(1-u)*(-b)+u*b
        if k in ('salt','pepper','mixed','uniform_impulse','gaussian_impulse'):
            p=c['rate'];pp=p if k=='pepper' else 0. if k=='salt' else p/2;ps=p if k=='salt' else 0. if k=='pepper' else p/2
            u=rng.random(a.shape);labels[u<pp]=-1;labels[(u>=pp)&(u<pp+ps)]=1;a[labels==-1]=0.;a[labels==1]=1.
    events=labels!=0
    if k in ('gaussian','uniform') and sd>0:events=np.ones(a.shape,bool)
    return a,labels,events


def audit(directory):
    directory=Path(directory);required={n+'.png' for n in FIGURES}|{n+'.csv' for n in GROUPS}|{'metrics.json','reference_arrays.npz','artifact_manifest.json'}
    require({p.name for p in directory.iterdir() if p.is_file()}==required,'File inventory')
    manifest=json.loads(portable.canonical(directory/'artifact_manifest.json'))
    require(len(manifest['files'])==len(required)-1 and {r['path'] for r in manifest['files']}==required-{'artifact_manifest.json'},'Manifest inventory')
    for r in manifest['files']:
        data=portable.canonical(directory/r['path']);require(len(data)==r['canonical_bytes'] and portable.digest(data)==r['sha256'],'Hash/size '+r['path'])
        require(r['policy']==('utf8-crlf-to-lf' if Path(r['path']).suffix in ('.json','.csv') else 'raw-binary'),'Integrity policy')
    for n in FIGURES:figure_io.verify_png(directory/(n+'.png'))
    p=json.loads(portable.canonical(directory/'metrics.json'));cases,tasks=design();expected_meta=[{k:v for k,v in c.items() if k!='clean'} for c in cases]
    require(p['schema']==1 and p['cases']==expected_meta and p['intensity_range']==1. and p['correction_tolerance']==.05 and p['false_alteration_tolerance']==1e-12 and p['structure_tolerance']==.1,'Parameters and conventions')
    require(p['figures']==list(FIGURES) and p['groups']==list(GROUPS),'Figure/table names')
    with np.load(directory/'reference_arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    require(len(a)==p['array_count'] and all(v.dtype.kind in 'fib' and np.isfinite(v).all() for v in a.values()),'Array inventory/finiteness')
    keys=set();mm=masks()
    for n,m in mm.items():
        key='mask_'+n;keys.add(key);require(a[key].dtype==np.bool_ and np.array_equal(a[key],m),'Structure masks')
    for c in cases:
        n=c['id'];noisy,labels,events=independent_observation(c)
        for s,v in dict(clean=c['clean'],noisy=noisy,noise=noisy-c['clean'],labels=labels,events=events).items():
            key=n+'_'+s;keys.add(key);require(a[key].shape==v.shape and a[key].dtype==v.dtype,'Source dtype/shape')
            require(np.array_equal(a[key],v) if v.dtype.kind in 'bi' else np.allclose(a[key],v,rtol=8*np.finfo(float).eps,atol=0),'Ground truth '+key)
        if n.startswith('density_'):
            key=n+'_counts';keys.add(key);counts=a[key];require(counts.dtype==np.int16 and counts.shape==events.shape,'Count dtype/shape')
            for y,x in np.ndindex(events.shape):require(counts[y,x]==sum(samples(events,y,x,(3,3))),'Local majority count')
    rows=p['rows'];require(len(rows)==len(tasks)+40,'Rows')
    count=0;maximum=relative=stress_abs=stress_rel=0.
    for i,(n,g,params) in enumerate(tasks):
        row=rows[i];key=f'result_{i:04d}';keys.add(key)
        for k,v in dict(id=key,case=n,group=g,**params,survivors=math.prod(params['window'])-params['d']).items():require(row[k]==v,'Row parameters')
        clean,noisy,events=(a[n+'_'+s] for s in ('clean','noisy','events'));out=a[key];h,w=clean.shape
        require(out.shape==clean.shape and out.dtype==np.float64,'Result dtype/shape')
        points=sorted({(0,0),(0,w-1),(h-1,0),(h-1,w-1),(h//2,w//2),(min(17,h-1),min(10,w-1)),(min(23,h-1),min(12,w-1)),(h//2-1,w//2-1),(h//3,w//3)})
        expected=reference(noisy,params['method'],params['window'],d=params['d'],mode=params['mode'],cval=params['cval'],points=points)
        selected=np.array([out[y,x] for y,x in points]);diff=abs(selected-expected)
        require(np.allclose(selected,expected,rtol=1e-12,atol=1e-14),'Independent reference '+key)
        if params['method'] in ('median','minimum','maximum'):require(np.array_equal(selected,expected),'Exact order statistic')
        require(np.array_equal(selected[expected==0],expected[expected==0]),'Exact zero')
        maximum=max(maximum,float(diff.max()));relative=max(relative,float((diff/np.maximum(abs(expected),np.finfo(float).tiny)).max()));count+=len(points)
        for k,v in independent_metrics(clean,noisy,out,events).items():close(row[k],v,'Metric '+k)
        if 'bright1_survival' in row:
            for name,mask in mm.items():close(row[name+'_survival'],avg((abs(out[mask]-clean[mask])<=.1).astype(float)),'Structure survival')
            close(row['profile_max_gradient'],max(abs(float(out[19,x+1]-out[19,x])) for x in range(w-1)),'Profile gradient')
            close(row['profile_rmse'],math.sqrt(avg((out[19]-clean[19])**2)),'Profile RMSE')
            sd=lambda v:0. if np.ptp(v)==0 else math.sqrt(avg((v-avg(v))**2))
            close(row['texture_sd_ratio'],sd(out[mm['texture']])/sd(clean[mm['texture']]),'Texture SD')
        if n.startswith('density_'):
            good=a[n+'_counts']<=4;close(row['majority_clean_fraction'],avg(good.astype(float)),'Majority fraction')
            close(row['good_majority_error_rate'],avg((abs(out[good]-.5)>.05).astype(float)),'Good majority error')
            close(row['bad_majority_correction_rate'],avg((abs(out[~good]-.5)<=.05).astype(float)),'Bad majority recovery')
            if params['method']=='median':require(np.all(out[good]==.5),'Clean local-majority guarantee')
    j=len(tasks)
    for n,image in stresses().items():
        key='stress_'+n;keys.add(key);require(np.array_equal(a[key],image),'Stress input')
        for m in METHODS[:-1]:
            key='stress_'+n+'_'+m;keys.update((key,key+'_reference'));r=rows[j];j+=1
            require(all(r[k]==v for k,v in dict(id=key,case=n,group='stress',method=m,window=[1,3],d=0,mode='reflect',cval=0.,survivors=3).items()),'Stress metadata')
            expected=reference(image,m,(1,3));out=a[key]
            require(out.shape==image.shape and out.dtype==np.float64 and a[key+'_reference'].dtype==np.float64,'Stress shape/dtype')
            require(np.array_equal(a[key+'_reference'],expected) and np.allclose(out,expected,rtol=1e-14,atol=0),'Stress reference')
            diff=abs(out-expected);rel=float((diff/np.maximum(abs(expected),np.finfo(float).tiny)).max())
            close(r['max_absolute_error'],float(diff.max()),'Stress absolute',atol=0);close(r['max_relative_error'],rel,'Stress relative',atol=0)
            stress_abs=max(stress_abs,float(diff.max()));stress_rel=max(stress_rel,rel)
    require(set(a)==keys,'Unexpected/missing arrays')
    for g in GROUPS:
        table=list(csv.DictReader(io.StringIO(portable.canonical(directory/(g+'.csv')).decode())))
        rr=[r for r in rows if r['group']==g];fields=set().union(*(r.keys() for r in rr))
        require(len(table)==len(rr) and set(table[0])==fields,'CSV schema/count')
        for entry,row in zip(table,rr):
            for k in fields:
                v=row.get(k);text=entry[k]
                if v is None:require(text=='','CSV null')
                elif isinstance(v,str):require(text==v,'CSV string')
                elif isinstance(v,list):require(json.loads(text)==v,'CSV list')
                else:require(float(text)==v,'CSV numerical coherence')
    return dict(scientific_checks='passed',arrays=len(a),rows=len(rows),figures=len(FIGURES),tables=len(GROUPS),independent_pixels=count,
        ordinary_max_absolute_error=maximum,ordinary_max_relative_error=relative,stress_max_absolute_error=stress_abs,stress_max_relative_error=stress_rel)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',nargs='?',type=Path,default=Path(__file__).resolve().parent/'output')
    print(json.dumps(audit(parser.parse_args().directory),indent=2,allow_nan=False))
