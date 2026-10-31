"""Reproducible synthetic band/notch experiments; algorithms import independently."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from notch_support import low_pass, figure_io, portable
from notch_filters import notch_reject, band_reject, notch_profile, band_profile, filter_image
from notch_scenes import make_scenes
from notch_metrics import evaluate, ringing
from peak_detection import detect_peaks

METHODS = ('noisy','ideal','butterworth','gaussian','radial_butterworth','lowpass')
KINDS = ('ideal','butterworth','gaussian')
RADIUS = .012


def responses(shape, components, radius=RADIUS, order=2):
    centers = [c['frequency_cpp'] for c in components]
    h = {'noisy':np.ones(shape)}
    for kind in KINDS:
        h[kind] = notch_reject(shape, centers, radius, kind, order=order)
    radial = np.ones(shape)
    for center in sorted({float(np.hypot(*p)) for p in centers}):
        radial *= band_reject(shape, center, 2*radius, 'butterworth', order=order)
    h['radial_butterworth']=radial
    h['lowpass']=low_pass.gaussian_low_pass(shape,.10)
    return h


def write_table(path, rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def panels(directory, name, items, *, columns=3, title='', shared=False):
    """Each item=(title, array, domain); spectrum axes are physical cpp."""
    rows=(len(items)+columns-1)//columns
    fig,axes=plt.subplots(rows,columns,figsize=(4.4*columns,3.4*rows),squeeze=False)
    fig.suptitle(title,fontsize=12)
    vmin=min(float(a.min()) for _,a,_ in items) if shared else None
    vmax=max(float(a.max()) for _,a,_ in items) if shared else None
    for ax,(label,a,domain) in zip(axes.flat,items):
        opts=dict(vmin=vmin,vmax=vmax) if shared else {}
        if domain=='frequency':
            m,n=a.shape;fy,fx=low_pass.frequency_axes(a.shape,centered=True)
            opts.update(origin='lower',extent=(fx[0]-.5/n,fx[-1]+.5/n,fy[0]-.5/m,fy[-1]+.5/m))
        image=ax.imshow(a,interpolation='nearest',cmap='viridis',**opts)
        ax.set_title(label,fontsize=9)
        ax.set_xlabel('column frequency [cycles/pixel]' if domain=='frequency' else 'column [pixel]')
        ax.set_ylabel('row frequency [cycles/pixel]' if domain=='frequency' else 'row [pixel]')
        unit = ('dimensionless gain' if 'gain' in label else 'ln(1+|F|)') if domain=='frequency' else ('signed intensity error' if 'signed error' in label else 'intensity')
        fig.colorbar(image,ax=ax,shrink=.8,label=unit)
    for ax in list(axes.flat)[len(items):]:ax.set_visible(False)
    fig.tight_layout();figure_io.save_figure(fig,directory/name);plt.close(fig)


def run(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    scenes=make_scenes();shape=scenes['vertical']['clean'].shape
    arrays={'clean':scenes['vertical']['clean']};outputs={};masks={};measurements=[];detections={};detection_rows=[]
    for name,scene in scenes.items():
        outputs[name]={};masks[name]=responses(shape,scene['components'])
        for field in ['noisy','periodic','gaussian']:
            key = 'gaussian_noise' if field=='gaussian' else field
            arrays[name+'_'+key]=scene[field]
        for method,h in masks[name].items():
            full=np.fft.ifft2(np.fft.fft2(scene['noisy'])*h)
            outputs[name][method]=full.real
            measurements.append(dict(scene=name,method=method,**evaluate(scene['clean'],scene['noisy'],scene['periodic'],scene['components'],h)))
            if method!='noisy':arrays[name+'_'+method]=full.real
        detections[name]=detect_peaks(scene['noisy'],max_pairs=3)
    detections['clean_only']=detect_peaks(arrays['clean'],max_pairs=3)
    for name,proposals in detections.items():
        truth=[] if name=='clean_only' else scenes[name]['components']
        for j,p in enumerate(proposals):
            f=np.array(p['frequency_cpp']);distances=[]
            for component in truth:
                for sign in [1,-1]:
                    d=np.abs(f-sign*np.array(component['frequency_cpp']));d=np.minimum(d,1-d)
                    distances.append(float(np.hypot(*d)))
            detection_rows.append(dict(scene=name,rank=j+1,row=p['index'][0],column=p['index'][1],
                                       fy_cpp=p['frequency_cpp'][0],fx_cpp=p['frequency_cpp'][1],
                                       partner_row=p['partner_index'][0],partner_column=p['partner_index'][1],
                                       magnitude=p['magnitude'],contrast_score=p['contrast_score'],
                                       matched_known_noise=bool(distances and min(distances)<.02)))
    for method in METHODS[1:]:arrays['response_'+method]=masks['vertical'][method]
    arrays['vertical_spectrum']=np.fft.fft2(scenes['vertical']['noisy'])
    parameters=[];width_items=[];order_results={}
    scene=scenes['vertical'];centers=[c['frequency_cpp'] for c in scene['components']]
    for radius in [.004,.012,.035]:
        for kind in KINDS:
            h=notch_reject(shape,centers,radius,kind)
            out=np.fft.ifft2(np.fft.fft2(scene['noisy'])*h).real
            parameters.append(dict(sweep='radius',kind=kind,radius=radius,order=2,
                                   **evaluate(scene['clean'],scene['noisy'],scene['periodic'],scene['components'],h),**ringing(h)))
            width_items.append((f'{kind}, r={radius} cpp',out,'spatial'))
    for order in [1,2,4,8]:
        h=notch_reject(shape,centers,RADIUS,'butterworth',order=order)
        order_results[order]=(h,np.fft.ifft2(np.fft.fft2(scene['noisy'])*h).real)
        parameters.append(dict(sweep='order',kind='butterworth',radius=RADIUS,order=order,
                               **evaluate(scene['clean'],scene['noisy'],scene['periodic'],scene['components'],h),**ringing(h)))
    radial=[]
    for kind in KINDS:
        for distance in np.linspace(0,.5,501):
            radial.append(dict(kind=kind,distance_cpp=float(distance),band_gain=float(band_profile(np.array(distance),.1875,.024,kind)),
                               single_notch_gain=float(notch_profile(np.array(distance),RADIUS,kind))))
    # Nonperiodic crop: preserve physical frequency centers under changed size.
    crop=np.s_[10:83,9:110]
    arrays['boundary_clean']=scene['clean'][crop];arrays['boundary_noisy']=scene['noisy'][crop]
    boundary_rows=[]
    for name,padding,boundary in [('periodic',(0,0),'reflect'),('reflect',(24,24),'reflect'),('zero',(24,24),'zero')]:
        result=filter_image(arrays['boundary_noisy'],centers=centers,radius=RADIUS,padding=padding,boundary=boundary)
        values=result.values.real;arrays['boundary_'+name]=values
        mse=float(np.mean((values-arrays['boundary_clean'])**2))
        boundary_rows.append(dict(mode=name,padding_rows=padding[0],padding_columns=padding[1],
                                  padded_rows=result.padded_shape[0],padded_columns=result.padded_shape[1],
                                  mse=mse,psnr_db=float(-10*np.log10(mse)),imaginary_residual=result.imaginary_residual))
    # High-confidence counterexample: clean legitimate periodic texture is removed.
    wrong=notch_reject(shape,[(18/96,0)],RADIUS)
    arrays['damaged_legitimate']=np.fft.ifft2(np.fft.fft2(arrays['clean'])*wrong).real
    auto_centers=[p['frequency_cpp'] for p in detections['clean_only']]
    auto_h=notch_reject(shape,auto_centers,RADIUS)
    arrays['automatic_clean']=np.fft.ifft2(np.fft.fft2(arrays['clean'])*auto_h).real
    noisy_auto_h=notch_reject(shape,[p['frequency_cpp'] for p in detections['vertical']],RADIUS)
    arrays['automatic_noisy']=np.fft.ifft2(np.fft.fft2(scene['noisy'])*noisy_auto_h).real
    step=np.broadcast_to((np.arange(shape[1])>=shape[1]//2).astype(float),shape).copy();arrays['step']=step
    for kind in KINDS:arrays['step_'+kind]=np.fft.ifft2(np.fft.fft2(step)*masks['vertical'][kind]).real
    failure=dict(legitimate_notch_mse=float(np.mean((arrays['damaged_legitimate']-arrays['clean'])**2)),
                 automatic_clean_mse=float(np.mean((arrays['automatic_clean']-arrays['clean'])**2)),
                 automatic_noisy_mse=float(np.mean((arrays['automatic_noisy']-arrays['clean'])**2)))
    # 1. Direct visual comparison with common absolute intensity range.
    panels(directory,'overview.png',[('Known clean',arrays['clean'],'spatial')]+[(m,outputs['vertical'][m],'spatial') for m in METHODS],columns=4,shared=True,title='Vertical interference; radius=.012 cpp, order=2; shared intensity scale')
    # 2. Annotated spectrum and explicit detector proposals.
    fig,axes=plt.subplots(1,2,figsize=(12,4.7));fy,fx=low_pass.frequency_axes(shape,centered=True)
    extent=(fx[0]-.5/shape[1],fx[-1]+.5/shape[1],fy[0]-.5/shape[0],fy[-1]+.5/shape[0])
    for ax,label,spectrum in zip(axes,['Noisy spectrum; red=true noise, cyan=legitimate texture','Optional detector proposals; not noise classifications'],[arrays['vertical_spectrum']]*2):
        im=ax.imshow(np.log1p(abs(np.fft.fftshift(spectrum))),extent=extent,origin='lower');ax.set(title=label,xlabel='fx [cycles/pixel]',ylabel='fy [cycles/pixel]');fig.colorbar(im,ax=ax,label='ln(1+|F|)')
    for sign in [-1,1]:
        axes[0].plot(sign*.1875,0,'rx',markersize=10);axes[0].plot(0,sign*.1875,'c+',markersize=10)
    for p in detections['vertical']:
        for key in ['frequency_cpp','partner_frequency_cpp']:
            fy0,fx0=p[key];axes[1].plot(fx0,fy0,'r+',markersize=9)
        axes[1].annotate(f"score={p['contrast_score']:.1f}",(p['frequency_cpp'][1],p['frequency_cpp'][0]),
                         xytext=(6,7),textcoords='offset points',fontsize=8,color='white',
                         bbox=dict(facecolor='black',alpha=.6,edgecolor='none'))
    fig.tight_layout();figure_io.save_figure(fig,directory/'spectral_peaks.png');plt.close(fig)
    # 3. Band response profiles and maps, not arbitrary bin-unit conversions.
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for kind,ax in zip(KINDS,axes.flat):
        h=band_reject(shape,.1875,.024,kind);im=ax.imshow(np.fft.fftshift(h),extent=extent,origin='lower',vmin=0,vmax=1);ax.set(title=kind+' band reject',xlabel='fx [cycles/pixel]',ylabel='fy [cycles/pixel]');fig.colorbar(im,ax=ax,label='gain')
    for kind in KINDS:
        rows=[r for r in radial if r['kind']==kind];axes[1,1].plot([r['distance_cpp'] for r in rows],[r['band_gain'] for r in rows],label=kind)
    axes[1,1].set(xlabel='D [cycles/pixel]',ylabel='gain',title='D0=.1875 cpp, W=.024 cpp, n=2');axes[1,1].legend();fig.tight_layout();figure_io.save_figure(fig,directory/'band_responses.png');plt.close(fig)
    # 4. Notch response maps.
    panels(directory,'notch_responses.png',[(k+' paired notch gain',np.fft.fftshift(masks['vertical'][k]),'frequency') for k in KINDS],title='Product of local factors; toroidal geometry; DC explicitly protected')
    # 5,6. Independent parameter axes.
    panels(directory,'width_sensitivity.png',width_items,shared=True,title='Radius sweep at known true frequencies; n=2; shared intensity scale')
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for ax,(order,(h,values)) in zip(axes.flat,order_results.items()):
        im=ax.imshow(values);ax.set(title=f'Butterworth n={order}, r=.012 cpp',xlabel='column [pixel]',ylabel='row [pixel]');fig.colorbar(im,ax=ax,label='intensity')
    axis=np.fft.fftfreq(shape[1]);positive=(axis>=0)&(axis<.35)
    for order,(h,_) in order_results.items():axes[1,1].plot(axis[positive],h[0,positive],label=f'n={order}')
    axes[1,1].set(xlabel='fx [cycles/pixel], fy=0',ylabel='paired gain',title='Exact sampled pair response');axes[1,1].legend();axes[1,2].axis('off');fig.tight_layout();figure_io.save_figure(fig,directory/'order_sensitivity.png');plt.close(fig)
    # 7,8. Orientation and simultaneous components.
    items=[]
    for name in ['horizontal','vertical','diagonal']:
        items.extend([(name+' noisy',scenes[name]['noisy'],'spatial'),(name+' Butterworth',outputs[name]['butterworth'],'spatial'),(name+' signed error',outputs[name]['butterworth']-arrays['clean'],'spatial')])
    panels(directory,'orientations.png',items,title='Stripes name spatial orientation; spectrum peaks follow variation direction')
    s=scenes['multiple'];out=outputs['multiple']['butterworth']
    panels(directory,'multiple_noise.png',[('Noisy',s['noisy'],'spatial'),('Known periodic noise',s['periodic'],'spatial'),('Butterworth correction',out,'spatial'),('ln(1+|shift F|)',np.log1p(abs(np.fft.fftshift(np.fft.fft2(s['noisy'])))),'frequency'),('Paired transfer gain',np.fft.fftshift(masks['multiple']['butterworth']),'frequency'),('Removed component',s['noisy']-out,'spatial')],title='Three simultaneous frequencies; unique conjugate pairs')
    # 9. Off-bin energy remains beyond a narrow rejection region.
    leakage=scenes['leakage'];narrow=outputs['leakage']['ideal'];broad_h=notch_reject(shape,[p['frequency_cpp'] for p in leakage['components']],.035,'butterworth');broad=np.fft.ifft2(np.fft.fft2(leakage['noisy'])*broad_h).real
    panels(directory,'leakage.png',[('Aligned periodic ln(1+|shift F|)',np.log1p(abs(np.fft.fftshift(np.fft.fft2(scene['periodic'])))),'frequency'),('Off-bin periodic ln(1+|shift F|)',np.log1p(abs(np.fft.fftshift(np.fft.fft2(leakage['periodic'])))),'frequency'),('Off-bin observation',leakage['noisy'],'spatial'),('Ideal r=.012 cpp',narrow,'spatial'),('Butterworth r=.035 cpp',broad,'spatial'),('Narrow ideal signed error',narrow-arrays['clean'],'spatial')],title='Leakage: fractional bins (10.35,22.4); broader rejection also damages signal')
    # 10. Counterexamples, boundary models, and signed ringing profiles.
    fig,axes=plt.subplots(3,3,figsize=(13,11))
    images=[('Known clean texture',arrays['clean']),('Wrong manual notch',arrays['damaged_legitimate']),('Automatic clean proposals applied',arrays['automatic_clean']),('Cropped periodic',arrays['boundary_periodic']),('Cropped reflect pad24',arrays['boundary_reflect']),('Cropped zero pad24',arrays['boundary_zero'])]
    for ax,(label,a) in zip(axes.flat,images):
        im=ax.imshow(a);ax.set(title=label,xlabel='column [pixel]',ylabel='row [pixel]');fig.colorbar(im,ax=ax,label='intensity')
    for kind in KINDS:axes[2,0].plot(arrays['step_'+kind][shape[0]//2],label=kind)
    axes[2,0].plot(step[0],'k--',label='true step');axes[2,0].set(xlabel='column [pixel]',ylabel='signed intensity',title='Step ringing; r=.012 cpp');axes[2,0].legend()
    for name in ['periodic','reflect','zero']:axes[2,1].plot(arrays['boundary_'+name][36]-arrays['boundary_clean'][36],label=name)
    axes[2,1].set(xlabel='cropped column [pixel]',ylabel='signed error',title='Boundary row 36; unchanged cpp centers');axes[2,1].legend()
    axes[2,2].bar(['wrong manual','automatic clean'],[failure['legitimate_notch_mse'],failure['automatic_clean_mse']]);axes[2,2].set(ylabel='MSE against clean',title='High confidence is not quality');fig.tight_layout();figure_io.save_figure(fig,directory/'failure_boundaries.png');plt.close(fig)
    # 11. Difficult amplitudes, near Nyquist and additional random noise.
    items=[]
    for name in ['high','weak','near_nyquist','gaussian_mixed']:
        items.extend([(name+' noisy',scenes[name]['noisy'],'spatial'),(name+' Butterworth',outputs[name]['butterworth'],'spatial')])
    panels(directory,'stress_cases.png',items,columns=2,title='No clipping or uint8 conversion; each colorbar reports actual range')
    # 12. Multiple metrics: PSNR alone cannot select a perceptual winner.
    fig,axes=plt.subplots(1,2,figsize=(14,5));names=list(scenes);positions=np.arange(len(names));width=.12
    for j,method in enumerate(METHODS):
        values=[next(r['psnr_db'] for r in measurements if r['scene']==n and r['method']==method) for n in names]
        axes[0].bar(positions+(j-2.5)*width,values,width=width,label=method)
    axes[0].set_xticks(positions,names,rotation=50,ha='right');axes[0].set(ylabel='PSNR [dB], reference range=1',title='Known clean truth; all ten scenes');axes[0].legend(fontsize=7)
    for method in METHODS:
        r=next(r for r in measurements if r['scene']=='vertical' and r['method']==method)
        axes[1].scatter(r['off_noise_clean_spectral_error'],r['periodic_rms_gain'],label=method)
    axes[1].legend(fontsize=8)
    axes[1].set(xlabel='Off-noise clean spectral relative error',ylabel='Periodic component RMS gain',title='Signal damage versus noise removal');fig.tight_layout();figure_io.save_figure(fig,directory/'quantitative.png');plt.close(fig)
    scene_specs={name:{k:scene[k] for k in ['components','gaussian_sigma','seed']} for name,scene in scenes.items()}
    metrics=dict(settings=dict(shape=list(shape),radius_cpp=RADIUS,order=2,baseline_cutoff_cpp=.10,seed=707,
                               detector=dict(dc_radius=.035,threshold=6.,max_pairs=3,separation=.025)),
                 scenes=scene_specs,measurements=measurements,parameters=parameters,detections=detections,
                 detection_rows=detection_rows,boundaries=boundary_rows,failure=failure,
                 clean_reference=dict(mse=0.,psnr_db=None,psnr_infinite=True),
                 array_inventory={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()})
    for name,rows in [('measurements',measurements),('parameters',parameters),('detections',detection_rows),('boundaries',boundary_rows),('radial_profiles',radial)]:write_table(directory/(name+'.csv'),rows)
    portable.write_json(directory/'metrics.json',metrics)
    np.savez_compressed(directory/'reference_arrays.npz',**arrays)
    portable.publish_manifest(directory)
    print(f'Generated 12 figures, {len(arrays)} arrays, {len(measurements)} measurements, {len(parameters)} parameter records, {len(detection_rows)} proposals.')
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'output')
    run(parser.parse_args().output_dir)
