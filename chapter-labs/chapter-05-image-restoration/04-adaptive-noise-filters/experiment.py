"""Reproducible runner; production algorithms are independently importable."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adaptive_filters import adaptive_local_noise_filter,adaptive_median_filter,estimate_noise_variance
from adaptive_reference import reference
from adaptive_design import design,observe,masks,stress_inputs,ROI,BAD_ROI,FIGURES,GROUPS
from adaptive_measurements import quality,diagnostic_counts
from adaptive_support import means,order,portable,figure_io


def apply(image,params):
    p=dict(params);method=p.pop('method')
    if method=='local':
        p.pop('variance_source');p.pop('factor')
        return adaptive_local_noise_filter(image,**p,return_diagnostics=True)
    if method=='adaptive_median':return adaptive_median_filter(image,**p,return_diagnostics=True)
    function=means.arithmetic_mean if method=='mean' else order.median_filter
    return function(image,**p),{}


def generate(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);cases,tasks=design();byid={c['id']:c for c in cases};a={};rows=[]
    for n,m in masks().items():a['mask_'+n]=m
    for c in cases:
        noisy,labels,events=observe(c);n=c['id']
        for s,v in dict(clean=c['clean'],noisy=noisy,noise=noisy-c['clean'],labels=labels,events=events).items():a[n+'_'+s]=v
    for i,(n,g,original) in enumerate(tasks):
        p=dict(original);clean,noisy,events=(a[n+'_'+s] for s in ('clean','noisy','events'));c=byid[n];truth=c['sigma']**2
        if p['method']=='local':
            source=p['variance_source']
            variance=estimate_noise_variance(noisy,BAD_ROI if source=='contaminated' else ROI) if source in ('estimated','contaminated') else .05**2 if source=='assumed_mismatch' else truth
            p['noise_variance']=variance*p['factor']
        key=f'result_{i:04d}';out,d=apply(noisy,p);a[key]=out
        for field,v in d.items():a[key+'__'+field]=v
        row=dict(id=key,case=n,group=g,params=p,**quality(clean,noisy,out,events))
        if p['method']=='local':row.update(variance_error=p['noise_variance']-truth,variance_relative_error=(p['noise_variance']/truth-1) if truth else None)
        if p['method']=='adaptive_median':row.update(diagnostic_counts(d))
        if clean.shape==(32,44):
            for name,mask in masks().items():
                row[name+'_mse']=float(np.mean((out[mask]-clean[mask])**2))
                row[name+'_survival']=float((abs(out[mask]-clean[mask])<=.1).mean())
        rows.append(row)
    # Small full-map scientific references, separate from ordinary quality metrics.
    for n,image in stress_inputs().items():
        a['stress_'+n]=image
        for method in ('local','adaptive_median'):
            p=dict(method='local',window=[1,3],regularization='clipped',variance_source='oracle',factor=1.,noise_variance=.0025,mode='reflect',cval=0.) if method=='local' else dict(method='adaptive_median',initial_size=3,max_size=5,mode='reflect',cval=0.)
            with np.errstate(all='raise'):out,d=apply(image,p)
            ref,rd=reference(image,method,p);key='stress_'+n+'_'+method;a[key]=out;a[key+'__reference']=ref.reshape(image.shape)
            for field,v in d.items():a[key+'__'+field]=v
            diff=abs(out-ref.reshape(image.shape));rows.append(dict(id=key,case=n,group='stress',params=p,max_absolute_error=float(diff.max()),max_relative_error=float((diff/np.maximum(abs(ref.reshape(image.shape)),np.finfo(float).tiny)).max())))
    payload=dict(schema=1,range=1.,roi=list(ROI),contaminated_roi=list(BAD_ROI),correction_tolerance=.05,alteration_tolerance=1e-12,structure_tolerance=.1,
        cases=[{k:v for k,v in c.items() if k!='clean'} for c in cases],rows=rows,array_count=len(a),figures=list(FIGURES),groups=list(GROUPS))
    portable.write_json(directory/'metrics.json',payload)
    for g in GROUPS:
        rr=[r for r in rows if r['group']==g];fields=sorted(set().union(*(r.keys() for r in rr)))
        with (directory/(g+'.csv')).open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
            for r in rr:w.writerow({k:json.dumps(v,sort_keys=True,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    np.savez_compressed(directory/'reference_arrays.npz',**a);figures(directory,a,payload);portable.publish_manifest(directory)
    return payload


def figures(directory,a,p):
    plt.rcParams.update({'font.size':9,'axes.titlesize':10});rows=p['rows'];cases={c['id']:c for c in p['cases']}
    def row(case,method,group='main',**kwargs):return next(r for r in rows if r['case']==case and r['group']==group and r['params']['method']==method and all(r['params'].get(k)==v for k,v in kwargs.items()))
    def save(fig,n,title):fig.suptitle(title,fontsize=12);figure_io.save_figure(fig,directory/(n+'.png'));plt.close(fig)
    def panels(n,title,items,cols=3):
        fig,axes=plt.subplots((len(items)+cols-1)//cols,cols,figsize=(4*cols,3.2*((len(items)+cols-1)//cols)),layout='constrained',squeeze=False)
        for ax,(label,key) in zip(axes.flat,items):
            im=ax.imshow(a[key],cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set(title=label,xlabel='column [pixel]',ylabel='row [pixel]')
        for ax in list(axes.flat)[len(items):]:ax.set_visible(False)
        fig.colorbar(im,ax=list(axes.flat),label='intensity [0,1], display saturation only',shrink=.7);save(fig,n,title)
    loc=row('medium','local',regularization='clipped',variance_source='oracle')['id']
    est=row('medium','local',variance_source='estimated')['id'];med=row('mixed','adaptive_median')['id']
    panels('overview','Original 32×44 composite; matched inputs, explicit noise models', [('Clean','medium_clean'),('Gaussian σ=.05','medium_noisy'),('Oracle clipped local, 7×7',loc),('Mixed impulses p=.20','mixed_noisy'),('Fixed median 3×3',row('mixed','fixed_median')['id']),('Adaptive median 3→7',med)])
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,key,label in zip(axes,(loc+'__mean',loc+'__variance',loc+'__weight'),('Local mean [intensity]','Population variance [intensity²]','Clipped ratio weight [dimensionless]')):
        im=ax.imshow(a[key],cmap='viridis');ax.set(title=label,xlabel='column [pixel]',ylabel='row [pixel]');fig.colorbar(im,ax=ax)
    save(fig,'local_statistics','7×7 NumPy reflect; variance computed about shifted local mean')
    panels('gaussian','Same 7×7 footprints: literal formula can overshoot',[(c+' / '+m,row(c,m,**({'regularization':'clipped','variance_source':'oracle'} if m=='local' else {}))['id']) for c in ('low','high') for m in ('mean','local')]+[('Medium / literal',row('medium','local',regularization='literal')['id']),('Medium / estimated',est)])
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for source in ('oracle','estimated','contaminated'):
        rr=[r for r in rows if r['group']=='calibration' and r['params']['method']=='local' and r['params']['variance_source']==source]
        axes[0].plot(range(3),[r['params']['noise_variance'] for r in rr],'-o',label=source)
        axes[1].plot(range(3),[r['mse'] for r in rr],'-o',label=source)
    # Composite scene contaminated ROI demonstrates signal-variance bias.
    r=row('medium','local','variance',variance_source='contaminated');axes[0].axhline(r['params']['noise_variance'],color='red',ls=':',label='textured composite ROI')
    axes[0].axhline(.05**2,color='black',ls='--',label='known σ²');axes[0].set(ylabel='variance [intensity²]',xlabel='trial index',title='Predeclared ROI, no clean input');axes[1].set(ylabel='MSE [intensity²]',xlabel='trial index',title='3 seeded flat fields')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    save(fig,'variance_estimates','ROI estimate assumes flatness; contamination is not observation noise')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for policy in ('clipped','literal'):
        rr=[r for r in rows if r['group']=='variance' and r['params']['variance_source']=='oracle' and r['params']['regularization']==policy]
        axes[0].plot([r['params']['factor'] for r in rr],[r['mse'] for r in rr],'-o',label=policy)
        axes[1].plot([r['params']['factor'] for r in rr],[r['bright_survival'] for r in rr],'-o',label=policy)
    axes[0].set_ylabel('MSE [intensity²]');axes[1].set_ylabel('true bright-line survival [fraction]')
    for ax in axes:ax.set_xlabel('assumed / known variance');ax.legend();ax.grid(alpha=.2)
    save(fig,'variance_sweep','Under/overestimation: clipped shrinkage differs from literal extrapolation')
    panels('details','Noiseless legitimate extremes can trigger adaptive-median replacement',[('Original clean','clean_clean'),('Adaptive median 3→7',row('clean','adaptive_median')['id']),('Fixed median 3×3',row('clean','fixed_median')['id']),('Nonuniform noise','nonuniform_noisy'),('Global known right-half variance',row('nonuniform','local',variance_source='oracle',regularization='clipped')['id']),('Estimated ROI in clean half',row('nonuniform','local',variance_source='estimated')['id'])])
    panels('median_comparison','Variable-window decisions always use the original noisy observation',[(c+' / '+m,c+'_noisy' if m=='noisy' else row(c,m)['id']) for c in ('salt','pepper','mixed') for m in ('noisy','fixed_median','adaptive_median')])
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,field in zip(axes.flat,('size','expansions','stage_a_passed','stage_b_replaced','retained','fallback')):
        im=ax.imshow(a[med+'__'+field],cmap='viridis',interpolation='nearest');ax.set(title=field,xlabel='column [pixel]',ylabel='row [pixel]');fig.colorbar(im,ax=ax)
    save(fig,'decisions','Mixed p=.20: selected window [pixel], expansions [count], decision maps [0/1]')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for maximum in (3,5,7,9):
        xx=[];yy=[];ss=[];fallback=[]
        for rate in (0.,.1,.3,.5,.7,.9,1.):
            rr=[r for r in rows if r['group']=='density' and r['params']['method']=='adaptive_median' and r['params']['max_size']==maximum and cases[r['case']]['rate']==rate]
            v=[r['mse'] for r in rr];xx.append(rate);yy.append(np.mean(v));ss.append(np.std(v,ddof=1));fallback.append(np.mean([r['fallbacks']/(24*32) for r in rr]))
        axes[0].errorbar(xx,yy,yerr=ss,marker='o',capsize=3,label='Smax='+str(maximum));axes[1].plot(xx,fallback,'-o',label='Smax='+str(maximum))
    axes[0].set_ylabel('MSE [intensity²], 3 seeds ±SD');axes[1].set_ylabel('fallback fraction (return original)')
    for ax in axes:ax.set_xlabel('total impulse probability');ax.legend();ax.grid(alpha=.2)
    save(fig,'max_size','Larger windows can help, but unresolved Stage A keeps the original impulse')
    panels('high_density','No valid interior median may exist before Smax; no unconditional recovery',[('Clean','severe_clean'),('Mixed p=.85','severe_noisy'),('Adaptive median 3→7',row('severe','adaptive_median')['id']),('Fixed median 3×3',row('severe','fixed_median')['id'])],2)
    panels('mismatch','Different mechanisms: local shrinkage on impulses, adaptive median on Gaussian',[('Impulse observation','mixed_noisy'),('Assumed additive variance .0025',row('mixed','local',variance_source='assumed_mismatch',regularization='clipped')['id']),('Gaussian observation','medium_noisy'),('Adaptive median on Gaussian',row('medium','adaptive_median')['id'])],2)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for method in ('mean','fixed_median','local','adaptive_median'):
        rr=[row(c,method,**({'regularization':'clipped','variance_source':'oracle'} if method=='local' else {})) for c in ('low','medium','high')]
        axes[0].plot((.015,.05,.12),[r['psnr_db'] for r in rr],'-o',label=method)
    axes[0].set(xlabel='Gaussian σ [intensity]',ylabel='PSNR [dB], range=1',title='Composite; PSNR does not describe detail loss');axes[0].legend(fontsize=8)
    rr=[r for r in rows if r['group']=='boundary'];xx=np.arange(len(rr));axes[1].bar(xx,[r['border_mse'] for r in rr],label='border');axes[1].plot(xx,[r['interior_mse'] for r in rr],'ko',label='interior')
    axes[1].set_xticks(xx,[r['params']['mode']+'\n'+r['params']['method'] for r in rr],rotation=20,fontsize=7);axes[1].set(ylabel='MSE [intensity²]',title='Boundary policy; fixed 2-pixel metric band');axes[1].legend()
    save(fig,'quantitative','Matched inputs and explicit boundary assumptions')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'output')
    p=generate(parser.parse_args().output_dir);print(json.dumps(dict(arrays=p['array_count'],rows=len(p['rows']),figures=len(FIGURES))))
