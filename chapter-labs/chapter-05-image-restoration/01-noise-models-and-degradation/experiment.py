"""Standalone deterministic noise/degradation laboratory; never denoises inputs."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from noise_models import Distribution, Sinusoid, periodic_noise, impulse_replace, spatial_blur, degrade
from noise_statistics import describe, histogram, ecdf, quantiles, quality, periodogram, directional_power
from noise_scenes import clean_image, SHAPE, SEED, PARAMETERS, PSF
from noise_support import figure_io, portable
import noise_reference as reference

FIGURES = 'degradation models pdfs gaussian_sweep impulse_sweep positive_distributions estimation spectra periodic blur equal_variance summary'.split()


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.titlesize':11, 'figure.dpi':120})
    # fftshift frequencies are centers; extents are half a bin beyond centers.
    extent = [-.5-.5/128, .5-.5/128, -.5-.5/96, .5-.5/96]
    arrays = {'clean':clean_image(), 'psf':PSF.copy()}
    metrics = {'settings':dict(seed=SEED, shape=list(SHAPE), data_range=1., ddof_population=0,
                              ddof_sample=1, boundary='periodic', psf_anchor=[1,1], display_range=[0,1]),
               'models':{}, 'sweeps':[], 'impulses':[], 'periodic':[], 'patches':[]}

    def save(fig, name):
        figure_io.save_figure(fig, output/(name+'.png'))
        plt.close(fig)

    def show(ax, data, title, *, vmin=0, vmax=1, cmap='gray'):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set(title=title, xlabel='column [pixel]', ylabel='row [pixel]')
        return im

    for index, (name, (a,b)) in enumerate(PARAMETERS.items()):
        law = Distribution(name,a,b)
        noise = law.sample(SHAPE, rng=np.random.default_rng(SEED+index))
        observed = arrays['clean']+noise
        sample = law.sample((256,256), rng=np.random.default_rng(1501+index))
        mean, variance = law.moments()
        measured = describe(sample)
        edges = np.linspace(float(sample.min()), float(sample.max()), 65)
        counts, density = histogram(sample, edges)
        values, ranks = ecdf(sample)
        power, fy, fx = periodogram(noise)
        py, px = directional_power(power)
        parts = dict(noise=noise, observed=observed, sample=sample, edges=edges, counts=counts,
                     density=density, ecdf_values=values, ecdf_p=ranks, quantiles=quantiles(sample,[.01,.25,.5,.75,.99]),
                     power=power, fy=fy, fx=fx, directional_y=py, directional_x=px)
        arrays.update({name+'_'+key:value for key,value in parts.items()})
        se_mean, se_var = reference.uncertainty(name,a,b,sample.size)
        metrics['models'][name] = dict(a=a,b=b,seed=SEED+index,sample_seed=1501+index,
                                      theory_mean=mean,theory_variance=variance,sample_statistics=measured,
                                      noise_statistics=describe(noise),observed_statistics=describe(observed),
                                      quality=quality(arrays['clean'],observed),mean_standard_error=se_mean,
                                      variance_standard_error=se_var,mean_error_SE=(measured['mean']-mean)/se_mean,
                                      variance_error_SE=(measured['sample_variance']-variance)/se_var)
    for sigma in [0.,.01,.025,.05,.1,.2]:
        noise = Distribution('gaussian',0.,sigma).sample(SHAPE,rng=np.random.default_rng(2501))
        arrays[f'gaussian_sweep_{len(metrics["sweeps"])}'] = noise
        metrics['sweeps'].append(dict(sigma=sigma,**describe(noise),**quality(arrays['clean'],arrays['clean']+noise)))
    for pp, ps in [(0.,0.),(0.,.08),(.08,0.),(.04,.04),(.1,.1),(.25,.25)]:
        observed, labels = impulse_replace(arrays['clean'],p_pepper=pp,p_salt=ps,rng=np.random.default_rng(3501))
        k = len(metrics['impulses'])
        arrays.update({f'impulse_{k}':observed,f'labels_{k}':labels,f'impulse_residual_{k}':observed-arrays['clean']})
        metrics['impulses'].append(dict(p_pepper=pp,p_salt=ps,seed=3501,
                                       pepper_fraction=float(np.mean(labels==-1)),salt_fraction=float(np.mean(labels==1)),
                                       **quality(arrays['clean'],observed)))
    specs = [('horizontal',[Sinusoid(12/96,0,.08,.3)]),
             ('vertical',[Sinusoid(0,20/128,.08,.3)]),
             ('diagonal',[Sinusoid(12/96,20/128,.08,.3)]),
             ('multiple',[Sinusoid(12/96,0,.06,.3),Sinusoid(0,20/128,.06,.7)]),
             ('leakage',[Sinusoid(12.4/96,20.3/128,.08,.3)]),
             ('near_nyquist',[Sinusoid(0,63/128,.08,.3)])]
    for name, components in specs:
        noise = periodic_noise(SHAPE,components)
        power, _, _ = periodogram(noise)
        arrays[name+'_periodic'], arrays[name+'_periodic_power'] = noise,power
        chosen = np.argsort(power.ravel(),kind='stable')[-6:][::-1]
        bins = [list(map(int,np.unravel_index(int(k),SHAPE))) for k in chosen]
        pairs = set()
        for c in components:
            k,l = int(np.rint(c.fy*96))%96,int(np.rint(c.fx*128))%128
            pairs.update([(k,l),((-k)%96,(-l)%128)])
        metrics['periodic'].append(dict(name=name,components=[dict(fy=c.fy,fx=c.fx,amplitude=c.amplitude,
                                          phase=c.phase,wavevector_angle_degrees=c.wavevector_angle_degrees) for c in components],
                                        top_bins=bins,nearest_pair_power_fraction=float(sum(power[k,l] for k,l in sorted(pairs))/power.sum()),
                                        statistics=describe(noise)))
    arrays['blur_periodic'] = spatial_blur(arrays['clean'],PSF)
    arrays['blur_zero'] = spatial_blur(arrays['clean'],PSF,boundary='zero')
    arrays['blur_noise'] = Distribution('gaussian',0.,.04).sample(SHAPE,rng=np.random.default_rng(4501))
    arrays['blur_observed'] = degrade(arrays['clean'],PSF,arrays['blur_noise'])
    arrays['identity_observed'] = degrade(arrays['clean'],[[1.]],np.zeros(SHAPE))
    origin = np.zeros(SHAPE)
    for i,j in np.ndindex(PSF.shape):
        origin[(i-1)%96,(j-1)%128] += PSF[i,j]
    arrays['psf_origin'] = origin
    arrays['fourier_observed'] = np.fft.fft2(arrays['blur_observed'])
    arrays['fourier_model'] = np.fft.fft2(origin)*np.fft.fft2(arrays['clean'])+np.fft.fft2(arrays['blur_noise'])
    metrics['degradation'] = dict(identity_max_error=float(np.max(abs(arrays['identity_observed']-arrays['clean']))),
                                 spatial_max_error=float(np.max(abs(arrays['blur_observed']-arrays['blur_periodic']-arrays['blur_noise']))),
                                 fourier_max_error=float(np.max(abs(arrays['fourier_model']-arrays['fourier_observed']))),
                                 blur_only_quality=quality(arrays['clean'],arrays['blur_periodic']),
                                 noise_only_quality=quality(arrays['clean'],arrays['clean']+arrays['blur_noise']),
                                 combined_quality=quality(arrays['clean'],arrays['blur_observed']))
    for name,ys,xs in [('flat',(8,28),(8,28)),('textured',(65,90),(70,120))]:
        sl = (slice(*ys),slice(*xs))
        c,o,eta = arrays['clean'][sl],arrays['gaussian_observed'][sl],arrays['gaussian_noise'][sl]
        metrics['patches'].append(dict(name=name,rows=list(ys),columns=list(xs),clean_variance=describe(c)['variance'],
                                      observed_variance=describe(o)['variance'],true_noise_variance=describe(eta)['variance'],
                                      mean_noise_estimate=describe(o)['mean']-describe(c)['mean']))

    fig,ax = plt.subplots(figsize=(11,3.4))
    ax.axis('off')
    for xpos,text in [(.08,'Ideal image f'),(.35,'PSF convolution h*f'),(.65,'+ additive noise eta'),(.91,'Observed g')]:
        ax.text(xpos,.55,text,ha='center',va='center',bbox=dict(boxstyle='round',fc='#edf2fa'),transform=ax.transAxes)
    for x0,x1 in [(.15,.25),(.45,.54),(.76,.85)]:
        ax.annotate('',xy=(x1,.55),xytext=(x0,.55),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    ax.text(.5,.13,'Periodic boundary: G = H F + N. Impulses replace pixels. Restoration is reserved for later labs.',ha='center',transform=ax.transAxes)
    save(fig,'degradation')
    fig,axs = plt.subplots(2,3,figsize=(12,7))
    show(axs.flat[0],arrays['clean'],'Clean synthetic image')
    for ax,name in zip(list(axs.flat)[1:],PARAMETERS):
        show(ax,arrays[name+'_observed'],name+' additive, no centering')
    fig.suptitle('Shared display [0,1]; measurements retain unclipped float64')
    fig.tight_layout();save(fig,'models')
    fig,axs = plt.subplots(2,3,figsize=(12,7))
    for ax,(name,(a,b)) in zip(axs.flat,PARAMETERS.items()):
        edges=arrays[name+'_edges'];centers=(edges[:-1]+edges[1:])/2
        ax.stairs(arrays[name+'_density'],edges,label='empirical, n=65536')
        ax.plot(centers,reference.distribution(name,a,b).pdf(centers),label='theoretical PDF')
        ax.set(title=name,xlabel='noise intensity',ylabel='density [1/intensity]');ax.legend(fontsize=8)
    axs.flat[-1].axis('off');fig.tight_layout();save(fig,'pdfs')
    fig,axs = plt.subplots(2,3,figsize=(12,7))
    for k,(ax,row) in enumerate(zip(axs.flat,metrics['sweeps'])):
        show(ax,arrays['clean']+arrays[f'gaussian_sweep_{k}'],f"sigma={row['sigma']}; MSE={row['mse']:.4g}")
    fig.suptitle('Same standardized draw at each Gaussian sigma; shared display [0,1]')
    fig.tight_layout();save(fig,'gaussian_sweep')
    fig,axs = plt.subplots(2,3,figsize=(12,7))
    for k,(ax,row) in enumerate(zip(axs.flat,metrics['impulses'])):
        show(ax,arrays[f'impulse_{k}'],f"pepper={row['p_pepper']}, salt={row['p_salt']}")
    fig.suptitle('Categorical replacement, fixed uniform draws; unselected pixels retained')
    fig.tight_layout();save(fig,'impulse_sweep')
    fig,axs = plt.subplots(1,3,figsize=(12,3.6))
    for ax,name in zip(axs,['rayleigh','erlang','exponential']):
        data=arrays[name+'_sample'];a,b=PARAMETERS[name];xx=np.linspace(0,float(data.max()),400)
        ax.hist(data.ravel(),bins=64,density=True,alpha=.4,label='samples')
        ax.plot(xx,reference.distribution(name,a,b).pdf(xx),label='PDF')
        ax.axvline(metrics['models'][name]['theory_mean'],color='black',ls='--',label='mean')
        ax.set(title=name,xlabel='noise intensity',ylabel='density');ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'positive_distributions')
    fig,axs = plt.subplots(1,3,figsize=(13,4));names=list(PARAMETERS)
    for ax,key in zip(axs[:2],['mean_error_SE','variance_error_SE']):
        ax.bar(names,[metrics['models'][n][key] for n in names]);ax.axhline(6,color='r',ls='--');ax.axhline(-6,color='r',ls='--')
        ax.set(ylabel='error / theoretical standard error',title=key.replace('_',' '));ax.tick_params(axis='x',rotation=25)
    x=np.arange(2);axs[2].bar(x-.15,[r['observed_variance'] for r in metrics['patches']],.3,label='observed patch')
    axs[2].bar(x+.15,[r['true_noise_variance'] for r in metrics['patches']],.3,label='actual noise')
    axs[2].set_xticks(x,['flat','textured']);axs[2].set(ylabel='population variance',title='Texture contaminates estimation');axs[2].legend(fontsize=8)
    fig.tight_layout();save(fig,'estimation')
    fig,axs = plt.subplots(2,3,figsize=(12,7))
    for col,name in enumerate(['gaussian','exponential','diagonal']):
        noise=arrays[name+'_noise'] if col<2 else arrays['diagonal_periodic']
        power=periodogram(noise)[0]
        show(axs[0,col],noise,name+' realization',vmin=-.12,vmax=.12,cmap='coolwarm')
        im=axs[1,col].imshow(10*np.log10(np.maximum(np.fft.fftshift(power),1e-12)),origin='lower',extent=extent,vmin=-100,vmax=20,cmap='magma')
        axs[1,col].set(xlabel='fx [cycles/pixel]',ylabel='fy [cycles/pixel]',title='Power [dB re 1 intensity²]');fig.colorbar(im,ax=axs[1,col],shrink=.7)
    fig.suptitle('Uncentered periodogram: nonzero mean adds DC; periodic noise is directional')
    fig.tight_layout();save(fig,'spectra')
    fig,axs = plt.subplots(2,6,figsize=(19,7))
    for col,row in enumerate(metrics['periodic']):
        name=row['name'];show(axs[0,col],arrays['clean']+arrays[name+'_periodic'],name)
        power=arrays[name+'_periodic_power']
        axs[1,col].imshow(10*np.log10(np.maximum(np.fft.fftshift(power),1e-12)),origin='lower',extent=extent,vmin=-100,vmax=20,cmap='magma')
        for c in row['components']:
            axs[1,col].plot([c['fx'],-c['fx']],[c['fy'],-c['fy']],'co',mfc='none',ms=7)
        axs[1,col].set(title=f"nearest-pair power {row['nearest_pair_power_fraction']:.3f}",xlabel='fx [cycles/pixel]',ylabel='fy [cycles/pixel]')
    fig.suptitle('Known frequency pairs (cyan): bin alignment, leakage and near-Nyquist behavior')
    fig.tight_layout();save(fig,'periodic')
    fig,axs = plt.subplots(2,3,figsize=(12,7))
    keys=['clean','blur_periodic','blur_zero','blur_noise','blur_observed','gaussian_observed']
    titles=['clean','blur: periodic boundary','blur: zero exterior','actual additive noise','periodic blur + noise','noise only']
    for ax,key,title in zip(axs.flat,keys,titles):
        show(ax,arrays[key],title,vmin=-.12 if key=='blur_noise' else 0,vmax=.12 if key=='blur_noise' else 1,cmap='coolwarm' if key=='blur_noise' else 'gray')
    fig.tight_layout();save(fig,'blur')
    fig,axs = plt.subplots(2,5,figsize=(17,7))
    for col,name in enumerate(PARAMETERS):
        show(axs[0,col],arrays[name+'_observed'][64:92,64:124],name+'; low-contrast detail')
        axs[1,col].hist(arrays[name+'_sample'].ravel(),bins=60,density=True);axs[1,col].set_xlim(-.16,.28)
        axs[1,col].set(xlabel='noise intensity',ylabel='density',title=f"mean={metrics['models'][name]['theory_mean']:.3f}; var=0.0016")
    fig.suptitle('Equal theoretical variance; different bias and tails; samples not standardized')
    fig.tight_layout();save(fig,'equal_variance')
    fig,axs = plt.subplots(1,3,figsize=(13,4))
    axs[0].bar(names,[metrics['models'][n]['quality']['mse'] for n in names]);axs[0].set(ylabel='MSE [intensity²]',title='Bias adds to error');axs[0].tick_params(axis='x',rotation=25)
    for name in names:
        axs[1].plot(arrays[name+'_ecdf_values'][::128],arrays[name+'_ecdf_p'][::128],label=name)
    axs[1].set(xlabel='noise intensity',ylabel='empirical CDF');axs[1].legend(fontsize=8)
    axs[2].axis('off');axs[2].text(0,.95,'Interpretation limits\n\nNo denoising here.\nDisplay saturation is not numerical clipping.\nIID moments do not describe periodic noise.\nTexture contaminates flat-patch estimation.\nPSNR alone is not perceptual quality.\nFinite samples differ from population values.',va='top',linespacing=1.5)
    fig.tight_layout();save(fig,'summary')

    np.savez_compressed(output/'reference_arrays.npz',**arrays)
    portable.write_json(output/'metrics.json',metrics)
    tables={'models':[dict(model=name,**{k:v for k,v in row.items() if not isinstance(v,dict)},
                            **{'sample_'+k:v for k,v in row['sample_statistics'].items()},**row['quality']) for name,row in metrics['models'].items()],
            'sweeps':metrics['sweeps'],'impulses':metrics['impulses'],'patches':metrics['patches']}
    for name,rows in tables.items():
        with (output/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
    portable.publish_manifest(output)
    print(json.dumps(dict(figures=len(FIGURES),arrays=len(arrays),tables={k:len(v) for k,v in tables.items()},degradation=metrics['degradation']),indent=2))
    return metrics


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'output')
    run(parser.parse_args().output_dir)
