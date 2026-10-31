"""Deterministic synthetic homomorphic experiments; run separately from imports."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from homomorphic import filter_image, transfer_function, radial_response
from homo_support import low_pass, figure_io
from homo_metrics import measure
from portable_artifacts import write_json, publish_manifest

DEFAULT = dict(gamma_low=.6, gamma_high=1.4, cutoff=.06, c=1.)


def synthetic_inputs(shape=(96, 128)):
    y, x = np.indices(shape, dtype=float)
    y /= shape[0]; x /= shape[1]
    rectangle = (x > .22) & (x < .72) & (y > .23) & (y < .75)
    r = np.exp(.12*np.cos(2*np.pi*(17*x+5*y))+.08*np.sin(2*np.pi*31*x)+.18*rectangle)
    illuminations = dict(uniform=np.full(shape,.7), gradient=np.exp(-1.3+1.6*x),
                         vignette=np.exp(-2.2*((x-.5)**2+(y-.5)**2)),
                         sinusoidal=np.exp(-.6+.7*np.cos(2*np.pi*x)+.35*np.sin(2*np.pi*y)),
                         dark=np.exp(-2.5+1.8*x))
    return r, illuminations


def save_panels(out, name, panels, columns=3, title=''):
    rows = (len(panels)+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.3*columns,3.5*rows), squeeze=False)
    fig.suptitle(title, fontsize=13)
    for ax, (label, data) in zip(axes.flat, panels):
        frequency_panel = 'shift FFT' in label or 'shift H' in label
        options = {}
        if frequency_panel:
            m,n=data.shape; fy=np.fft.fftshift(np.fft.fftfreq(m));fx=np.fft.fftshift(np.fft.fftfreq(n))
            options=dict(origin='lower',extent=(fx[0]-.5/n,fx[-1]+.5/n,fy[0]-.5/m,fy[-1]+.5/m))
        image = ax.imshow(data, cmap='viridis', interpolation='nearest', **options)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('column frequency [cycles/pixel]' if frequency_panel else 'column [pixel]')
        ax.set_ylabel('row frequency [cycles/pixel]' if frequency_panel else 'row [pixel]')
        units = ('intensity' if label.startswith('exp(') else
                 'log-domain gain' if 'shift H' in label else
                 'natural log amplitude' if 'log' in label else 'intensity')
        fig.colorbar(image, ax=ax, shrink=.8, label=units)
    for ax in list(axes.flat)[len(panels):]: ax.set_visible(False)
    fig.tight_layout(); figure_io.save_figure(fig, out/name); plt.close(fig)


def table(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def run(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':9, 'axes.grid':False})
    truth, illuminations = synthetic_inputs()
    arrays = {'reflectance':truth}; measurements=[]; outputs={}; checks=[]
    for name, illumination in illuminations.items():
        observed = truth*illumination
        result = filter_image(observed, **DEFAULT)
        outputs[name] = result.values
        arrays.update({f'illum_{name}':illumination, f'observed_{name}':observed, f'out_{name}':result.values})
        identity = filter_image(observed, gamma_low=1, gamma_high=1)
        lowpass = np.fft.ifft2(np.fft.fft2(observed)*low_pass.gaussian_low_pass(observed.shape,.06)).real
        methods = dict(input=observed, global_power=observed**.6, ordinary_lowpass=lowpass,
                       identity=identity.values, homomorphic=result.values)
        for method, values in methods.items():
            measurements.append(dict(scene=name, method=method, **measure(values,truth),
                                     imaginary_residual=result.imaginary_residual if method=='homomorphic' else 0.))
        checks.extend([dict(name=f'identity_{name}', max_absolute_error=float(np.max(abs(identity.values-observed)))),
                       dict(name=f'factorization_{name}',max_absolute_error=float(np.max(abs(np.log(observed)-np.log(truth)-np.log(illumination)))))])
        if name=='gradient':
            arrays.update(baseline_tone=methods['global_power'],baseline_lowpass=lowpass,baseline_identity=identity.values)
    observed = arrays['observed_gradient']
    response = transfer_function(observed.shape)
    arrays.update(log_input=np.log(observed), log_spectrum=np.fft.fft2(np.log(observed)),
                  response=response, filtered_log=np.log(outputs['gradient']))
    for gamma in [.6,1.,1.4]:
        value=filter_image(np.full((7,8),.7),gamma_low=gamma,gamma_high=1.8).values
        checks.append(dict(name=f'constant_{gamma}',max_absolute_error=float(np.max(abs(value-.7**gamma)))))
    checks.append(dict(name='global_scale_q3.2', max_absolute_error=float(np.max(abs(filter_image(observed*3.2).values-3.2**.6*outputs['gradient'])))))
    mode=.2*np.cos(2*np.pi*9*np.arange(128)/128)[None,:]+np.zeros((96,1))
    expected=np.exp(radial_response(np.array(9/128)) * mode)
    checks.append(dict(name='analytic_log_mode',max_absolute_error=float(np.max(abs(filter_image(np.exp(mode)).values-expected)))))
    parameters=[]; gamma_panels=[]; transition_panels=[]
    for lo in [.3,.6,1.]:
        for hi in [1.,1.4,2.]:
            result=filter_image(observed,gamma_low=lo,gamma_high=hi)
            parameters.append(dict(gamma_low=lo,gamma_high=hi,cutoff=.06,c=1.,**measure(result.values,truth),imaginary_residual=result.imaginary_residual))
            gamma_panels.append((f'gL={lo}, gH={hi}',result.values))
    for c in [.5,2.]:
        for cutoff in [.03,.06,.12]:
            result=filter_image(observed,c=c,cutoff=cutoff)
            parameters.append(dict(gamma_low=.6,gamma_high=1.4,cutoff=cutoff,c=c,**measure(result.values,truth),imaginary_residual=result.imaginary_residual))
            transition_panels.append((f'c={c}, D0={cutoff} cycles/pixel',result.values))
    radial=[]
    for c in [.5,1.,2.]:
        for d in np.linspace(0,.5,501):
            radial.append(dict(c=c,radius_cpp=float(d),gain=float(radial_response(np.array(d),c=c))))
    reflect=filter_image(observed,padding=(32,32)); arrays['out_reflect']=reflect.values
    rng=np.random.default_rng(606); noise=rng.normal(size=truth.shape)
    arrays['noise_bright']=arrays['observed_dark']+.004*noise
    arrays['out_noisy_bright']=filter_image(arrays['noise_bright']).values
    noise_rows=[]; noise_panels=[]
    for j,scale in enumerate([1.,1e-4,1e-5]):
        clean=arrays['observed_dark']*scale; raw=clean+2e-6*noise
        noisy=filter_image(raw,floor=1e-7,gamma_high=1.8)
        control=filter_image(clean,floor=1e-7,gamma_high=1.8)
        rms=float(np.sqrt(np.mean((raw-clean)**2)))
        noise_rows.append(dict(scale=scale,floor=1e-7,raw_noise_rms=rms,
                               output_delta_rms=float(np.sqrt(np.mean((noisy.values-control.values)**2))),
                               noise_gain=float(np.sqrt(np.mean((noisy.values-control.values)**2))/rms),
                               log_delta_rms=float(np.sqrt(np.mean((noisy.log_values-control.log_values)**2))),
                               floored_pixels=noisy.floored_pixels,imaginary_residual=noisy.imaginary_residual))
        arrays[f'raw_sensor_{j}']=raw; arrays[f'noise_output_{j}']=noisy.values
        noise_panels.extend([(f'input scale={scale:g}',raw),(f'output; floored={noisy.floored_pixels}',noisy.values)])
    save_panels(output,'decomposition.png',[(n,a) for n,a in [('Known reflectance r',truth),('Known illumination i',illuminations['gradient']),('Observed f=i*r',observed)]],title='Multiplicative model; 96 x 128, square pixels')
    save_panels(output,'pipeline.png', [('log f',np.log(observed)),('log(1+|shift FFT(log f)|)',np.log1p(abs(np.fft.fftshift(arrays['log_spectrum'])))),('shift H',np.fft.fftshift(response)),('inverse log',arrays['filtered_log']),('exp(inverse log)',outputs['gradient']),('Known reflectance',truth)],title='Log → FFT → H → inverse FFT → exponential; default parameters')
    save_panels(output,'gamma_sweep.png',gamma_panels,title='Gain sweep; D0=.06 cycles/pixel, c=1; individual color ranges')
    save_panels(output,'transition_sweep.png',transition_panels,title='Transition sweep; gammaL=.6, gammaH=1.4; individual ranges')
    save_panels(output,'corrections.png',sum(([(f'{n}: observed',arrays[f'observed_{n}']),(f'{n}: homomorphic',outputs[n])] for n in illuminations),[]),columns=2,title='Five known illumination fields; same reflectance')
    save_panels(output,'baselines.png',[('Observed',observed),('Known reflectance',truth),('Global power .6',arrays['baseline_tone']),('Intensity Gaussian LP .06 cpp',arrays['baseline_lowpass']),('Identity gains 1/1',arrays['baseline_identity']),('Homomorphic .6/1.4',outputs['gradient'])],title='Baselines have different purposes; smoothing is not illumination recovery')
    save_panels(output,'noise_sensitivity.png',noise_panels,columns=2,title='Same seeded additive noise, sigma=2e-6; explicit floor=1e-7')
    save_panels(output,'boundaries.png',[('Periodic',outputs['gradient']),('Reflect padding 32 each end',reflect.values),('Absolute boundary-model difference',abs(reflect.values-outputs['gradient']))],title='Reflection before log; padded shape 160 x 192, original crop retained')
    fig,ax=plt.subplots(figsize=(8,4))
    for c in [.5,1.,2.]:
        subset=[r for r in radial if r['c']==c];ax.plot([r['radius_cpp'] for r in subset],[r['gain'] for r in subset],label=f'c={c}')
    ax.axvline(.06,color='gray',linestyle='--',label='D0=.06 cpp');ax.set(xlabel='Radial frequency [cycles/pixel]',ylabel='Log-domain gain H',title='gammaL=.6, gammaH=1.4; H(D0) depends on c');ax.legend();fig.tight_layout();figure_io.save_figure(fig,output/'response.png');plt.close(fig)
    # Shared ROI display range permits an absolute dark-detail comparison.
    fig,axes=plt.subplots(1,3,figsize=(12,3.6)); roi=np.s_[30:65,10:55]
    dark=[arrays['observed_dark'][roi],outputs['dark'][roi],truth[roi]]; vmin=min(a.min() for a in dark);vmax=max(a.max() for a in dark)
    for ax,a,label in zip(axes,dark,['Dark observation','Homomorphic','Known reflectance']):
        im=ax.imshow(a,vmin=vmin,vmax=vmax);ax.set(title=label,xlabel='ROI column [pixel]',ylabel='ROI row [pixel]')
    fig.colorbar(im,ax=axes.tolist(),label='Intensity; shared range');figure_io.save_figure(fig,output/'dark_detail.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for method,a in [('input',observed),('power',arrays['baseline_tone']),('homomorphic',outputs['gradient']),('truth',truth)]:
        offset=(np.log(a)-np.log(truth)).mean(); axes[0].plot(np.exp(np.log(a[48])-offset),label=method)
    axes[0].set(xlabel='Column [pixel]',ylabel='Globally log-scale-aligned intensity',title='Row 48; alignment uses known truth');axes[0].legend()
    selected=[r for r in measurements if r['scene']=='gradient'];axes[1].bar(range(len(selected)),[r['log_reflectance_rmse'] for r in selected]);axes[1].set_xticks(range(len(selected)),[r['method'] for r in selected],rotation=35,ha='right');axes[1].set(ylabel='Aligned log RMSE',title='Gradient scene; lower is better');fig.tight_layout();figure_io.save_figure(fig,output/'quantitative.png');plt.close(fig)
    metrics=dict(settings=dict(shape=list(truth.shape),seed=606,defaults=DEFAULT,padding=[32,32]),measurements=measurements,parameters=parameters,noise=noise_rows,checks=checks,boundary_max_difference=float(np.max(abs(reflect.values-outputs['gradient']))))
    for name,rows in [('measurements.csv',measurements),('parameters.csv',parameters),('noise.csv',noise_rows),('radial.csv',radial)]:table(output/name,rows)
    write_json(output/'metrics.json',metrics);np.savez_compressed(output/'reference_arrays.npz',**arrays);publish_manifest(output)
    print(f'Generated {len(arrays)} arrays, 11 figures, {len(measurements)} baseline records, {len(parameters)} parameter records.')
    print('Maximum mathematical check error:',max(r['max_absolute_error'] for r in checks))
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'output')
    run(parser.parse_args().output_dir)
