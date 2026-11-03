"""Deterministic mean-filter experiments; all scientific arrays retained."""
import argparse,csv,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mean_filters as filters
from mean_scenes import make_scenes,clean_image,step_image,SHAPE,METHODS,Q_VALUES,WINDOWS,IMPULSE_SCENES,DETAILS,STRESS
from mean_support import figure_io,portable,noise
from mean_measurements import quality,details,edge_metrics
from mean_reference import decimal_mean

FIGURES='overview arithmetic_geometric harmonic q_sweep wrong_q mixed window_sizes profiles boundaries noise_sweep stress quantitative'.split()
TABLES=['main','q_sweep','windows','boundaries','noise_sweep','stress']


def apply(image,method,window=(3,3),*,Q=1.5,mode='reflect',cval=0.):
    if method=='contra':return filters.contraharmonic_mean(image,Q,window,mode=mode,cval=cval)
    return getattr(filters,method+'_mean')(image,window,mode=mode,cval=cval)


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    scenes=make_scenes();arrays={'positive_clean':clean_image(),'black_clean':scenes['black_salt']['clean'],'step_clean':step_image(),'calibration_clean':np.full((32,40),.55)}
    data=dict(settings=dict(shape=list(SHAPE),default_window=[3,3],default_mode='reflect',default_Q=1.5,data_range=1.,line_threshold_resolution=1e-12,
                            q_values=Q_VALUES,windows=[list(w) for w in WINDOWS],sweep_seeds=[601,602,603],sweep_sigmas=[0.,.01,.03,.06],details=DETAILS,
                            zero_policy='geometric/harmonic/negative-Q: any zero -> continuous limit zero; positive-Q all-zero -> zero'),
              scenes={},main=[],q_sweep=[],windows=[],boundaries=[],noise_sweep=[],stress=[])
    def row(scene,method,result,key,window=(3,3),Q=1.5,mode='reflect',cval=0.):
        s=scenes[scene]
        return dict(scene=scene,method=method,Q=Q if method=='contra' else None,window_h=window[0],window_w=window[1],mode=mode,cval=cval,output_key=key,
                    **quality(s['clean'],s['observed'],result,s['mask']),**details(s['clean'],result))
    for name,scene in scenes.items():
        for field in ['observed','noise','mask','labels']:arrays[name+'_'+field]=scene[field]
        data['scenes'][name]=dict(seed=scene['seed'],parameters=scene['parameters'],clean_key='black_clean' if name.startswith('black_') else 'positive_clean',
                                  corruption_fraction=float(np.mean(scene['mask'])),minimum_observed=float(scene['observed'].min()))
        for method in METHODS:
            key=name+'_'+method;arrays[key]=apply(scene['observed'],method)
            data['main'].append(row(name,method,arrays[key],key))
    for name in IMPULSE_SCENES:
        for index,Q in enumerate(Q_VALUES):
            key=f'q_{name}_{index}';arrays[key]=apply(scenes[name]['observed'],'contra',Q=Q)
            data['q_sweep'].append(row(name,'contra',arrays[key],key,Q=Q))
    for index,window in enumerate(WINDOWS):
        for method in METHODS:
            key=f'window_{index}_{method}';arrays[key]=apply(scenes['gaussian']['observed'],method,window)
            clean_key=key+'_clean';edge_key=key+'_edge'
            arrays[clean_key]=apply(arrays['positive_clean'],method,window);arrays[edge_key]=apply(arrays['step_clean'],method,window)
            entry=row('gaussian',method,arrays[key],key,window)
            entry.update(clean_output_key=clean_key,edge_output_key=edge_key,clean_mse=float(np.mean((arrays[clean_key]-arrays['positive_clean'])**2)),**edge_metrics(arrays[edge_key]))
            data['windows'].append(entry)
    border=np.ones(SHAPE,bool);border[3:-3,3:-3]=False;arrays['border_mask']=border
    for mode in ['reflect','edge','constant']:
        for label,method,Q in [('arithmetic','arithmetic',1.5),('geometric','geometric',1.5),('harmonic','harmonic',1.5),('contra','contra',1.5),('contra_negative','contra',-1.5)]:
            key=f'boundary_{mode}_{label}';arrays[key]=apply(scenes['gaussian']['observed'],method,(7,7),Q=Q,mode=mode,cval=0.)
            entry=row('gaussian',method,arrays[key],key,(7,7),Q,mode)
            error=arrays[key]-arrays['positive_clean']
            entry.update(label=label,border_mse=float(np.mean(error[border]**2)),interior_mse=float(np.mean(error[~border]**2)),border_bias=float(np.mean(error[border])))
            data['boundaries'].append(entry)
    for level,sigma in enumerate([0.,.01,.03,.06]):
        for seed in [601,602,603]:
            eta=noise.Distribution('gaussian',0.,sigma).sample((32,40),rng=np.random.default_rng(seed))
            observed=arrays['calibration_clean']+eta
            if (observed<=0).any():raise ValueError('Calibration is not positive; no hidden clipping')
            prefix=f'sweep_{level}_{seed}';arrays[prefix+'_noise']=eta;arrays[prefix+'_observed']=observed
            for method in METHODS:
                key=prefix+'_'+method;arrays[key]=apply(observed,method)
                error=(arrays[key]-arrays['calibration_clean'])[1:-1,1:-1]
                data['noise_sweep'].append(dict(sigma=sigma,seed=seed,method=method,output_key=key,
                                                **quality(arrays['calibration_clean'],observed,arrays[key],eta!=0),
                                                interior_residual_variance=float(np.var(error)),theory_arithmetic_variance=sigma*sigma/9))
    for case,values,Q in STRESS:
        source=np.array([values]);arrays['stress_'+case+'_input']=source
        for method in METHODS:
            key='stress_'+case+'_'+method;arrays[key]=apply(source,method,(1,len(values)),Q=Q)
            actual=float(arrays[key][0,len(values)//2])
            limit=method=='contra' and abs(Q)>1e100
            expected=float(max(values) if Q>0 else min(values)) if limit else decimal_mean(values,method,Q)
            data['stress'].append(dict(case=case,method=method,Q=Q if method=='contra' else None,output_key=key,
                                       actual=actual,reference=expected,reference_kind='concentration_limit' if limit else 'decimal100',
                                       absolute_error=abs(actual-expected),relative_error=abs(actual-expected)/abs(expected) if expected else 0.,ulp_error=abs(actual-expected)/math.ulp(expected)))
    plt.rcParams.update({'font.size':10,'axes.titlesize':10,'figure.dpi':120})
    def save(fig,name):figure_io.save_figure(fig,output/(name+'.png'));plt.close(fig)
    def image(ax,a,title):
        im=ax.imshow(a,cmap='gray',vmin=0,vmax=1);ax.set(title=title,xlabel='column [pixel]',ylabel='row [pixel]');return im
    fig,axs=plt.subplots(2,3,figsize=(12,7));image(axs.flat[0],arrays['positive_clean'],'Clean synthetic scene');image(axs.flat[1],arrays['gaussian_observed'],'Gaussian sigma=.025')
    for ax,method in zip(list(axs.flat)[2:],METHODS):image(ax,arrays['gaussian_'+method],method+(' Q=1.5' if method=='contra' else ''))
    fig.suptitle('3×3 NumPy reflect (edge excluded); shared display [0,1]; no numerical clipping');fig.tight_layout();save(fig,'overview')
    fig,axs=plt.subplots(2,4,figsize=(14,7))
    for rowno,name in enumerate(['gaussian','uniform']):
        for ax,key,title in zip(axs[rowno],[name+'_observed',name+'_arithmetic',name+'_geometric',name+'_harmonic'],[name,'arithmetic','geometric','harmonic']):image(ax,arrays[key],title)
    fig.suptitle('Positive additive-noise cases; nonlinear means can introduce bias');fig.tight_layout();save(fig,'arithmetic_geometric')
    fig,axs=plt.subplots(2,4,figsize=(14,7))
    for rowno,name in enumerate(['salt','black_salt']):
        for ax,key,title in zip(axs[rowno],[name+'_observed',name+'_arithmetic',name+'_geometric',name+'_harmonic'],[name,'arithmetic','geometric','harmonic']):image(ax,arrays[key],title)
    fig.suptitle('Salt suppression versus damage to genuine dark lines and black pixels');fig.tight_layout();save(fig,'harmonic')
    fig,axs=plt.subplots(2,7,figsize=(20,7))
    for rowno,name in enumerate(['salt','pepper']):
        for index,Q in enumerate(Q_VALUES):image(axs[rowno,index],arrays[f'q_{name}_{index}'],f'{name}; Q={Q}')
    fig.suptitle('Explicit zero limits; positive Q favors large samples, negative Q small samples');fig.tight_layout();save(fig,'q_sweep')
    fig,axs=plt.subplots(2,3,figsize=(12,7))
    for rowno,name in enumerate(['pepper','black_salt']):
        image(axs[rowno,0],arrays[name+'_observed'],name+' input')
        image(axs[rowno,1],arrays[f'q_{name}_0'],'Q=-1.5: zeros can spread')
        image(axs[rowno,2],arrays[f'q_{name}_6'],'Q=+1.5: bright-feature bias')
    fig.suptitle('Q mismatch is a failure case; real structure is not an impulse label');fig.tight_layout();save(fig,'wrong_q')
    fig,axs=plt.subplots(2,4,figsize=(14,7))
    for rowno,name in enumerate(['mixed','black_mixed']):
        for ax,key,title in zip(axs[rowno],[name+'_observed',name+'_arithmetic',f'q_{name}_0',f'q_{name}_6'],[name,'arithmetic','Q=-1.5','Q=+1.5']):image(ax,arrays[key],title)
    fig.suptitle('No fixed Q is guaranteed to remove both salt and pepper');fig.tight_layout();save(fig,'mixed')
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for method in METHODS:
        rows=[r for r in data['windows'] if r['method']==method and r['window_h']==r['window_w']]
        sizes=[r['window_h'] for r in rows]
        for ax,key in zip(axs,['mse','edge_width_10_90','clean_mse']):ax.plot(sizes,[r[key] for r in rows],'o-',label=method)
    for ax,title,unit in zip(axs,['Noisy-image error','Isolated clean-step width','Noiseless detail destruction'],['MSE [intensity²]','10–90% width [pixel]','MSE [intensity²]']):
        ax.set(xlabel='square window side [pixel]',ylabel=unit,title=title);ax.grid(alpha=.2)
    axs[0].legend(fontsize=8);fig.tight_layout();save(fig,'window_sizes')
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    axs[0].plot(np.arange(80),arrays['step_clean'][32],color='k',label='clean')
    axs[1].plot(np.arange(31,42),arrays['positive_clean'][31:42,20],color='k',label='clean')
    axs[2].plot(np.arange(39,50),arrays['positive_clean'][39:50,60],color='k',label='clean')
    index=WINDOWS.index((5,5))
    for method in METHODS:
        axs[0].plot(np.arange(80),arrays[f'window_{index}_{method}_edge'][32],label=method)
        axs[1].plot(np.arange(31,42),arrays[f'window_{index}_{method}_clean'][31:42,20],label=method)
        axs[2].plot(np.arange(39,50),arrays[f'window_{index}_{method}_clean'][39:50,60],label=method)
    axs[0].set(xlim=(34,46),xlabel='column [pixel]',ylabel='intensity',title='Clean step; 5×5')
    axs[1].set(xlabel='row [pixel]',ylabel='intensity',title='Thin dark line; 5×5');axs[2].set(xlabel='row [pixel]',ylabel='intensity',title='Thin bright line; 5×5')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();save(fig,'profiles')
    fig,axs=plt.subplots(1,3,figsize=(14,4));labels=['arithmetic','geometric','harmonic','contra','contra_negative'];x=np.arange(5)
    for offset,mode in enumerate(['reflect','edge','constant']):
        rows=[r for r in data['boundaries'] if r['mode']==mode]
        axs[0].bar(x+(offset-1)*.25,[r['border_mse'] for r in rows],.25,label=mode)
        axs[1].plot(x,[r['interior_mse'] for r in rows],'o-',label=mode)
    for ax in axs[:2]:ax.set_xticks(x,labels,rotation=30);ax.legend(fontsize=8)
    axs[0].set(ylabel='MSE [intensity²]',title='Border band (3 pixels)');axs[1].set(ylabel='MSE [intensity²]',title='Interior is boundary independent')
    for mode in ['reflect','edge','constant']:axs[2].plot(arrays[f'boundary_{mode}_harmonic'][0],label=mode)
    axs[2].plot(arrays['positive_clean'][0],'k--',label='clean');axs[2].set(xlabel='column [pixel]',ylabel='intensity',title='Harmonic top row; constant cval=0');axs[2].legend(fontsize=8)
    fig.tight_layout();save(fig,'boundaries')
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for method in METHODS:
        levels=[.01,.03,.06];summary=[];spread=[];bias=[]
        for sigma in levels:
            rows=[r for r in data['noise_sweep'] if r['method']==method and r['sigma']==sigma]
            summary.append(np.mean([r['mse'] for r in rows]));spread.append(np.std([r['mse'] for r in rows],ddof=1));bias.append(np.mean([r['residual_bias'] for r in rows]))
        axs[0].errorbar(levels,summary,yerr=spread,marker='o',label=method)
        axs[1].plot(levels,bias,'o-',label=method)
        vals=[np.mean([r['interior_residual_variance'] for r in data['noise_sweep'] if r['method']==method and r['sigma']==sigma]) for sigma in levels]
        axs[2].plot(levels,vals,'o-',label=method)
    axs[2].plot([.01,.03,.06],[s*s/9 for s in [.01,.03,.06]],'k--',label='arithmetic sigma²/9')
    for ax in axs:ax.set_xlabel('Gaussian sigma [intensity]');ax.legend(fontsize=7);ax.grid(alpha=.2)
    axs[0].set(ylabel='MSE [intensity²]',title='Three seeds: mean ± sample SD');axs[1].set(ylabel='mean residual [intensity]',title='Nonlinear estimator bias');axs[2].set(ylabel='variance [intensity²]',title='Flat interior; overlapping windows')
    fig.tight_layout();save(fig,'noise_sweep')
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    cases=[c[0] for c in STRESS]
    for method in METHODS:
        rows=[r for r in data['stress'] if r['method']==method]
        axs[0].semilogy(range(len(cases)),[max(r['relative_error'],1e-18) for r in rows],'o-',label=method)
    axs[0].set_xticks(range(len(cases)),cases,rotation=35);axs[0].set(ylabel='relative error (display floor 1e-18)',title='Independent 100-digit Decimal / limit oracles');axs[0].legend(fontsize=8)
    axs[1].axis('off');axs[1].text(0,.95,'Numerical contract\n\nNo products or reciprocal infinities.\nNo clipping or hidden epsilon.\nNegative intensities: arithmetic / Q=0 only.\nAny zero + negative Q: limit is exactly zero.\nAll zero + positive Q: limit is zero.\nNear-unity logs use log1p.\nNo extra longdouble precision assumed.',va='top',linespacing=1.5)
    fig.tight_layout();save(fig,'stress')
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    names=['gaussian','uniform','salt','pepper','mixed'];x=np.arange(5)
    for k,method in enumerate(METHODS):
        rows=[next(r for r in data['main'] if r['scene']==name and r['method']==method) for name in names]
        axs[0].bar(x+(k-1.5)*.19,[r['psnr_db'] for r in rows],.19,label=method)
        salt=next(r for r in rows if r['scene']=='salt')
        axs[1].scatter(salt['mse_uncorrupted'],salt['mse_corrupted'],label=method,s=45)
    axs[0].set_xticks(x,names,rotation=25);axs[0].set(ylabel='PSNR [dB], data range=1',title='Global quality is not universal ranking');axs[0].legend(fontsize=7)
    axs[1].set(xlabel='MSE on originally clean pixels',ylabel='MSE on salt pixels',title='Removal versus collateral damage');axs[1].legend(fontsize=8)
    axs[2].axis('off');axs[2].text(0,.95,'Interpretation limits\n\nMean filters destroy thin structures.\nGeometric is not always less blurring.\nNegative Q also favors genuine dark pixels.\nPositive Q can favor real bright objects.\nNo Q guarantees mixed-impulse recovery.\nDisplay [0,1] saturation is visualization only.',va='top',linespacing=1.5)
    fig.tight_layout();save(fig,'quantitative')
    np.savez_compressed(output/'reference_arrays.npz',**arrays);portable.write_json(output/'metrics.json',data)
    for table in TABLES:
        rows=data[table]
        with (output/(table+'.csv')).open('w',encoding='utf-8',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    portable.publish_manifest(output)
    print(json.dumps(dict(figures=len(FIGURES),arrays=len(arrays),tables={t:len(data[t]) for t in TABLES}),indent=2))
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'output');run(parser.parse_args().output_dir)
