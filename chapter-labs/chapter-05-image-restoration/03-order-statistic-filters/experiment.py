"""Deterministic runner; importable algorithms are in a separate module."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from order_design import METHODS,FIGURES,GROUPS,design,masks,observation,stresses,parameters
from order_statistic_filters import order_filter
from order_reference import reference
from order_measurements import quality,details
from order_support import means,figure_io,portable


def apply(a,p):
    return means.arithmetic_mean(a,p['window'],mode=p['mode'],cval=p['cval']) if p['method']=='arithmetic' else order_filter(a,**p)


def generate(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);cases,tasks=design();a={};rows=[];metadata=[]
    for n,m in masks().items():a['mask_'+n]=m
    for c in cases:
        n=c['id'];clean=c['clean'];noisy,labels,events=observation(c)
        for s,v in dict(clean=clean,noisy=noisy,noise=noisy-clean,labels=labels,events=events).items():a[n+'_'+s]=v
        if n.startswith('density_'):a[n+'_counts']=sliding_window_view(np.pad(events,1,mode='reflect'),(3,3)).sum(axis=(-1,-2)).astype(np.int16)
        metadata.append({k:v for k,v in c.items() if k!='clean'})
    for i,(n,g,p) in enumerate(tasks):
        clean,noisy,events=(a[n+'_'+s] for s in ('clean','noisy','events'));key=f'result_{i:04d}';out=apply(noisy,p);a[key]=out
        row=dict(id=key,case=n,group=g,**p,survivors=int(np.prod(p['window']))-p['d'],**quality(clean,noisy,out,events))
        if clean.shape==(40,56) and not n.startswith('density_'):row.update(details(clean,out,masks()))
        if n.startswith('density_'):
            good=a[n+'_counts']<=4
            row.update(majority_clean_fraction=float(good.mean()),good_majority_error_rate=float((abs(out[good]-.5)>.05).mean()) if good.any() else None,
                bad_majority_correction_rate=float((abs(out[~good]-.5)<=.05).mean()) if (~good).any() else None)
        rows.append(row)
    for n,image in stresses().items():
        a['stress_'+n]=image
        for m in METHODS[:-1]:
            p=parameters(m,(1,3),d=0);key='stress_'+n+'_'+m
            with np.errstate(all='raise'):out=apply(image,p)
            ref=reference(image,m,(1,3));a[key]=out;a[key+'_reference']=ref
            diff=abs(out-ref)
            rows.append(dict(id=key,case=n,group='stress',**p,survivors=3,max_absolute_error=float(diff.max()),max_relative_error=float((diff/np.maximum(abs(ref),np.finfo(float).tiny)).max())))
    payload=dict(schema=1,intensity_range=1.,correction_tolerance=.05,false_alteration_tolerance=1e-12,structure_tolerance=.1,cases=metadata,rows=rows,array_count=len(a),figures=list(FIGURES),groups=list(GROUPS))
    portable.write_json(directory/'metrics.json',payload)
    for group in GROUPS:
        rr=[r for r in rows if r['group']==group];fields=sorted(set().union(*(r.keys() for r in rr)))
        with (directory/(group+'.csv')).open('w',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
            for r in rr:writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()})
    np.savez_compressed(directory/'reference_arrays.npz',**a);figures(directory,a,payload);portable.publish_manifest(directory)
    return payload


def figures(directory,a,p):
    plt.rcParams.update({'font.size':9,'axes.titlesize':10})
    rows=p['rows'];cases={c['id']:c for c in p['cases']}
    def row(c,m,g='main',**kwargs):return next(r for r in rows if r['case']==c and r['method']==m and r['group']==g and all(r.get(k)==v for k,v in kwargs.items()))
    def save(fig,n,title):fig.suptitle(title,fontsize=12);figure_io.save_figure(fig,directory/(n+'.png'));plt.close(fig)
    def panels(n,title,items,cols=3):
        fig,axes=plt.subplots((len(items)+cols-1)//cols,cols,figsize=(4*cols,3.2*((len(items)+cols-1)//cols)),layout='constrained',squeeze=False)
        for ax,(label,key) in zip(axes.flat,items):
            im=ax.imshow(a[key],cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set(title=label,xlabel='column [pixel]',ylabel='row [pixel]')
        for ax in list(axes.flat)[len(items):]:ax.set_visible(False)
        fig.colorbar(im,ax=list(axes.flat),label='intensity [0,1], display only',shrink=.7);save(fig,n,title)
    panels('overview','3×3 reflect; mixed p=.20; original full-precision arrays retained',[('Clean','mixed_clean'),('Noisy, seed 734','mixed_noisy')]+[(m,row('mixed',m)['id']) for m in ('median','minimum','maximum','alpha_trimmed')])
    panels('median_impulses','Median: moderate noise versus majority-contamination failures',[(c+' / '+m,c+'_noisy' if m=='noisy' else row(c,'median')['id']) for c in ('mixed','high') for m in ('noisy','median')],2)
    panels('min_max','Salt/pepper asymmetry and legitimate-extrema propagation',[(c+' / '+m,row(c,m)['id']) for c in ('salt','pepper') for m in ('minimum','maximum','median')])
    panels('mean_comparison','Same 3×3 reflect: published arithmetic baseline versus order statistics',[(c+' / '+m,row(c,m)['id']) for c in ('gaussian','mixed') for m in ('arithmetic','median','alpha_trimmed')])
    panels('combined','Additive noise followed by categorical impulse replacement',[(c+' / '+m,row(c,m)['id']) for c in ('uniform_impulse','gaussian_impulse') for m in ('arithmetic','median','alpha_trimmed')])
    panels('windows','Identical scenes, different median window widths',[(c+' / '+str(w)+'×'+str(w),row(c,'median','windows',window=[w,w])['id']) for c in ('mixed','noiseless') for w in (3,5,7)])
    panels('boundaries','5×5, controlled corners; constant cval=−.2 saturates only in display',[(mode+' / '+m,row('border',m,'boundaries',mode=mode)['id']) for mode in ('reflect','edge','constant') for m in ('median','midpoint')],2)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,kind in zip(axes,('uniform','gaussian')):
        for m in ('midpoint','median','arithmetic'):
            yy=[];ss=[]
            for sd in (.01,.04,.08):
                v=[r['mse'] for r in rows if r['group']=='calibration' and r['method']==m and cases[r['case']]['kind']==kind and cases[r['case']]['sigma']==sd]
                yy.append(np.mean(v));ss.append(np.std(v,ddof=1))
            ax.errorbar((.01,.04,.08),yy,yerr=ss,marker='o',capsize=3,label=m)
        ax.set(xlabel='population σ [intensity]',ylabel='MSE [intensity²]',title=kind+'; 4 seeds, ±sample SD');ax.legend();ax.grid(alpha=.2)
    save(fig,'midpoint_uniform','Equal-variance flat fields: bounded uniform extrema versus Gaussian tails')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for c in ('gaussian','uniform','mixed','uniform_impulse','gaussian_impulse','high'):
        rr=[r for r in rows if r['group']=='trim' and r['case']==c]
        for ax,field in zip(axes,('mse','bright1_survival')):ax.plot([r['d'] for r in rr],[r[field] for r in rr],'-o',label=c)
    for ax in axes:ax.set_xlabel('total removed d; N=9, retain 9−d');ax.legend(fontsize=7);ax.grid(alpha=.2)
    axes[0].set_ylabel('MSE [intensity²]');axes[1].set_ylabel('bright 1-pixel line survival [fraction]');save(fig,'trim_sweep','Trimming: d=0 mean, d=8 median; no universal optimum')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for m in METHODS:
        v=a[row('noiseless',m,'windows',window=[3,3])['id']]
        axes[0].plot(v[19],label=m);axes[1].plot(np.arange(21,39),v[21:39,12],label=m)
    axes[0].plot(a['noiseless_clean'][19],'k--',label='truth');axes[1].plot(np.arange(21,39),a['noiseless_clean'][21:39,12],'k--',label='truth')
    axes[0].set(xlabel='column [pixel]',ylabel='intensity',title='Step profile, row 19');axes[1].set(xlabel='row [pixel]',ylabel='intensity',title='One/two-pixel bright/dark lines')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    save(fig,'details','Noiseless detail loss is not noise removal')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    pp=(0.,.1,.3,.5,.7,.9,1.)
    for m in ('median','alpha_trimmed','arithmetic'):
        yy=[];ss=[]
        for rate in pp:
            v=[r['correction_rate'] for r in rows if r['group']=='density' and r['method']==m and cases[r['case']]['rate']==rate];v=[x for x in v if x is not None]
            yy.append(np.mean(v) if v else np.nan);ss.append(np.std(v,ddof=1) if len(v)>1 else 0.)
        axes[0].errorbar(pp,yy,yerr=ss,marker='o',capsize=3,label=m)
    rr=[r for r in rows if r['group']=='density' and r['method']=='median']
    axes[1].scatter([cases[r['case']]['rate'] for r in rr],[r['majority_clean_fraction'] for r in rr],label='≤4 impulses / 9 samples')
    axes[0].set(xlabel='total impulse probability',ylabel='event correction |error|≤.05',title='3 seeds, ±sample SD; p=0 undefined');axes[1].set(xlabel='total impulse probability',ylabel='clean local-majority fraction',title='Local guarantee, no global deterministic threshold')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'density_failure','Flat truth .5; at p=1 odd median cannot recover .5')
    fig,ax=plt.subplots(figsize=(11,4.3),layout='constrained');cc=('gaussian','uniform','salt','pepper','mixed','uniform_impulse')
    v=np.array([[row(c,m)['psnr_db'] for c in cc] for m in METHODS]);im=ax.imshow(v,cmap='viridis',aspect='auto')
    ax.set_xticks(range(6),cc,rotation=20);ax.set_yticks(range(6),METHODS)
    for y,x in np.ndindex(v.shape):ax.text(x,y,f'{v[y,x]:.1f}',ha='center',va='center',color='white' if v[y,x]<23 else 'black')
    fig.colorbar(im,ax=ax,label='PSNR [dB], declared range 1');save(fig,'quantitative','Composite scenes: PSNR alone does not measure structural preservation')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'output')
    v=generate(p.parse_args().output_dir);print(json.dumps(dict(rows=len(v['rows']),arrays=v['array_count'],figures=len(FIGURES))))
