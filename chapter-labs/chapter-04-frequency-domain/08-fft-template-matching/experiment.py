"""Deterministic practical matching scenes, figures and separate runtime data."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from template_matching import match_template,fft_correlation,circular_correlation,zncc
from matching_scenes import make_scenes,geometric_template,sweep_scene
from matching_support import figure_io,portable
from benchmark import measure as benchmark

METHODS=['direct','fft','zncc','ssd']
FIGURES=['localization','direct_fft','padding_geometry','short_padding','ranking','affine','noise_sweep','ambiguity','occlusion_distractors','transformation_limits','size_comparison','quantitative']


def table(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def summary(name,method,result,locations,template_shape):
    info=result.diagnostics
    error=None if not locations or result.location is None else float(min(np.hypot(result.location[0]-y,result.location[1]-x) for y,x in locations))
    true_scores=[float(result.response[y,x]) for y,x in locations]
    return dict(scene=name,method=method,row=result.location[0] if result.location else None,column=result.location[1] if result.location else None,
                score=result.score,second_score=info['peaks'][1]['score'] if len(info['peaks'])>1 else None,
                true_score=max(true_scores) if true_scores else None,error_pixels=error,wrong_location=bool(error>.5) if error is not None else None,
                eligible_count=result.eligible_count,tie_count=info['tie_count'],ambiguous=info['ambiguous'],gap=info['gap'],psr=info['psr'],psr_status=info['psr_status'],
                imaginary_residual=result.numerical.imaginary_residual,fallback_windows=result.numerical.fallback_windows,
                response_rows=result.response.shape[0],response_columns=result.response.shape[1],template_rows=template_shape[0],template_columns=template_shape[1])


def panel(ax,a,title,unit='intensity',valid=None):
    if valid is not None:a=np.ma.masked_where(~valid,a)
    image=ax.imshow(a,interpolation='nearest',cmap='coolwarm' if unit=='ZNCC' else 'viridis',vmin=-1 if unit=='ZNCC' else None,vmax=1 if unit=='ZNCC' else None)
    ax.set(title=title,xlabel='column [pixel]',ylabel='row [pixel]')
    plt.colorbar(image,ax=ax,shrink=.78,label=unit)


def save(out,name,items,columns=3,title=''):
    fig,axes=plt.subplots((len(items)+columns-1)//columns,columns,figsize=(4.5*columns,3.5*((len(items)+columns-1)//columns)),squeeze=False)
    fig.suptitle(title,fontsize=12)
    for ax,item in zip(axes.flat,items):panel(ax,*item)
    for ax in list(axes.flat)[len(items):]:ax.set_visible(False)
    fig.tight_layout();figure_io.save_figure(fig,out/(name+'.png'));plt.close(fig)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    scenes=make_scenes();arrays={};measurements=[];results={};specs={}
    for name,s in scenes.items():
        for field in ['clean','image','template']:arrays[name+'_'+field]=s[field]
        specs[name]={k:s[k] for k in ['locations','seed','sigma','present','description']}
        specs[name].update(image_shape=list(s['image'].shape),template_shape=list(s['template'].shape))
        results[name]={}
        for method in METHODS:
            r=match_template(s['image'],s['template'],method=method);results[name][method]=r
            arrays[name+'_'+method]=r.response
            measurements.append(summary(name,method,r,s['locations'],s['template'].shape))
        arrays[name+'_valid']=results[name]['zncc'].validity
    a=scenes['unique']['image'];t=scenes['unique']['template'];base=results['unique']['zncc']
    arrays['affine_positive']=zncc(a*1.7+.4,t).values;arrays['affine_negative']=zncc(-a*1.7+.4,t).values
    sweeps=[];sweep_specs=[]
    for kind,values in [('noise',[0.,.04,.12,.24,.4,.7]),('occlusion',[0.,.2,.4,.6,.8]),('contrast',[-1.5,-.5,.25,1.,2.,4.])]:
        for j,value in enumerate(values):
            image,template,location=sweep_scene(kind,value);key=f'{kind}_{j}'
            arrays[key+'_image']=image;sweep_specs.append(dict(key=key,kind=kind,value=value,location=location))
            for method in METHODS:
                r=match_template(image,template,method=method)
                sweeps.append(dict(kind=kind,value=value,**summary(key,method,r,[location],template.shape)))
                if method=='zncc':arrays[key+'_zncc']=r.response;arrays[key+'_valid']=r.validity
    # Separate controlled absent trials; a score threshold is not a universal detector.
    trials=[]
    for j in range(20):
        image=.08+np.random.default_rng(9000+j).normal(0,.03,(48,64))
        category='background' if j<10 else 'near_distractor'
        if j>=10:image[20:29,30:41]=t+.08*np.random.default_rng(10000+j).normal(size=t.shape)
        r=match_template(image,t)
        trials.append(dict(seed=9000+j,category=category,score=r.score,row=r.location[0],column=r.location[1],threshold=.8,accepted=bool(r.score>=.8)))
    # Small geometry: full convolution, circular lags, and one-short aliases.
    g=np.arange(1.,36.).reshape(5,7)/35;kernel=np.array([[.2,.9],[-.4,.1],[.7,-.2]])
    full=fft_correlation(g,kernel);circ=circular_correlation(g,kernel)
    arrays.update(geometry_image=g,geometry_template=kernel,geometry_full=full.full,geometry_valid=full.values,geometry_circular=circ)
    p,q=full.full.shape
    for key,shape in [('short_row',(p-1,q)),('short_column',(p,q-1))]:
        arrays[key]=np.fft.ifft2(np.fft.fft2(g,s=shape)*np.fft.fft2(kernel[::-1,::-1],s=shape)).real
    wrapped=np.zeros((17,23));wy,wx=14,19
    for i,j in np.ndindex(t.shape):wrapped[(wy+i)%17,(wx+j)%23]=t[i,j]
    arrays['wrapped_image']=wrapped;arrays['wrapped_circular']=circular_correlation(wrapped,t);arrays['wrapped_valid']=fft_correlation(wrapped,t).values
    # Course-sized example uses new synthetic geometry, never the course image.
    course_t=geometric_template((38,42));course=np.zeros((256,256));course[218:256,214:256]=course_t
    course_r=fft_correlation(course,course_t)
    arrays.update(course_image=course,course_template=course_t,course_full=course_r.full,course_valid=course_r.values)
    course_info=dict(image_shape=[256,256],template_shape=[38,42],minimum_fft_shape=list(course_r.fft_shape),valid_shape=list(course_r.values.shape),location=list(map(int,np.unravel_index(np.argmax(course_r.values),course_r.values.shape))),imaginary_residual=course_r.imaginary_residual)
    # 1: localization with actual top-left coordinates.
    fig,axes=plt.subplots(1,3,figsize=(14,4));s=scenes['unique'];r=results['unique']['zncc']
    panel(axes[0],s['image'],'Known placement and detected box');y,x=s['locations'][0];h,w=t.shape
    axes[0].add_patch(Rectangle((x-.5,y-.5),w,h,fill=False,edgecolor='lime',linewidth=2));axes[0].set_title(f'GT={y,x}; detected={r.location}')
    panel(axes[1],t,'Asymmetric template');panel(axes[2],r.response,'Valid top-left scores','ZNCC',r.validity)
    axes[2].plot(r.location[1],r.location[0],'kx');fig.tight_layout();figure_io.save_figure(fig,out/'localization.png');plt.close(fig)
    # 2: raw maps share numerical limits; only error gets its own scale.
    fig,axes=plt.subplots(1,3,figsize=(14,4));direct=arrays['unique_direct'];fft=arrays['unique_fft'];lo=min(direct.min(),fft.min());hi=max(direct.max(),fft.max())
    for ax,data,label in zip(axes[:2],[direct,fft],['Direct spatial','FFT linear']):
        im=ax.imshow(data,vmin=lo,vmax=hi);ax.set(title=label,xlabel='top-left column [pixel]',ylabel='top-left row [pixel]');fig.colorbar(im,ax=ax,label='raw inner product [intensity²]')
    panel(axes[2],fft-direct,f'Max |difference|={np.max(abs(fft-direct)):.2e}','signed error [intensity²]');fig.tight_layout();figure_io.save_figure(fig,out/'direct_fft.png');plt.close(fig)
    save(out,'padding_geometry',[(g,'Image 5 x 7','intensity'),(kernel,'Template 3 x 2','intensity'),(full.full,'Full convolution: crop [2:5,1:7]','intensity²'),(full.values,'Valid C: 3 x 6, top-left lags','intensity²'),(wrapped,'Split object across image boundary','intensity'),(arrays['wrapped_circular'].real,'Circular peak can be outside valid lags','intensity²')],title='Unshifted coordinates; full convolution indices differ from valid top-left coordinates')
    save(out,'short_padding',[(full.full,'Minimal full 7 x 8','intensity²'),(arrays['short_row'],'One row short: 6 x 8','intensity²'),(arrays['short_row']-full.full[:-1],'Aliased top row; valid crop still correct','intensity²'),(arrays['short_column'],'One column short: 7 x 7','intensity²'),(arrays['short_column']-full.full[:,:-1],'Aliased left column','intensity²'),(full.values,'Strict API requires full padding','intensity²')],title='Full-map corruption does not imply corruption of every valid-only value')
    s=scenes['brightness_decoy'];items=[(s['image'],'True affine object, similar decoy, bright flat decoy','intensity')]
    for method in ['fft','ssd','zncc']:
        r=results['brightness_decoy'][method];items.append((r.response,f'{method}: best={r.location}', 'ZNCC' if method=='zncc' else 'score [intensity²]',r.validity))
    save(out,'ranking',items,columns=2,title='Different objective functions give different rankings; SSD is negated for maximization')
    save(out,'affine',[(base.response,'Original ZNCC','ZNCC',base.validity),(arrays['affine_positive'],'Image -> 1.7 I + .4','ZNCC',base.validity),(arrays['affine_positive']-base.response,'Positive affine score error','signed score error'),(arrays['affine_negative'],'Image -> -1.7 I + .4','ZNCC',base.validity),(arrays['affine_negative']+base.response,'Negative transform: score reversal error','signed score error'),(scenes['affine']['image'],'Positive transform of inserted patch only','intensity')],title='Positive affine invariance; negative contrast reverses sign, not absolute score')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for method in METHODS:
        rows=[r for r in sweeps if r['kind']=='noise' and r['method']==method];axes[0].plot([r['value'] for r in rows],[r['error_pixels'] for r in rows],'o-',label=method)
    for kind in ['noise','occlusion']:
        rows=[r for r in sweeps if r['kind']==kind and r['method']=='zncc'];axes[1].plot([r['value'] for r in rows],[r['true_score'] for r in rows],'o-',label=kind)
    axes[0].set(xlabel='Gaussian sigma [intensity]',ylabel='Localization error [pixel]',title='Fixed noise field; seed 1808');axes[0].legend()
    axes[1].set(xlabel='Noise sigma or requested occlusion fraction (legend)',ylabel='True-location ZNCC',title='Score decline does not define a detection probability');axes[1].legend();fig.tight_layout();figure_io.save_figure(fig,out/'noise_sweep.png');plt.close(fig)
    s=scenes['repeated'];r=results['repeated']['zncc'];fig,axes=plt.subplots(1,2,figsize=(12,4));panel(axes[0],s['image'],'Two identical inserted objects');panel(axes[1],r.response,f"ties={r.diagnostics['tie_count']}, gap={r.diagnostics['gap']}",'ZNCC',r.validity)
    for p in r.diagnostics['peaks']:axes[1].plot(p['column'],p['row'],'kx')
    fig.suptitle('Row-major numerical ties; top-2 after rectangular NMS');fig.tight_layout();figure_io.save_figure(fig,out/'ambiguity.png');plt.close(fig)
    items=[]
    for name in ['distractors','occlusion','absent']:
        s=scenes[name];r=results[name]['zncc'];items.extend([(s['image'],name,'intensity'),(r.response,f'best={r.location}, score={r.score:.3f}','ZNCC',r.validity)])
    save(out,'occlusion_distractors',items,columns=2,title='A best candidate always exists when valid windows exist; object presence is a separate decision')
    items=[]
    for name in ['rotated','scaled','near_constant','low_texture']:
        s=scenes[name];r=results[name]['zncc'];items.extend([(s['image'],name,'intensity'),(r.response,f'best={r.location}; eligible={r.eligible_count}','ZNCC',r.validity)])
    save(out,'transformation_limits',items,columns=2,title='Translation-only model; masked constant windows are shown blank, never eligible zero scores')
    save(out,'size_comparison',[(course,'New synthetic 256 x 256 image','intensity'),(course_t,'Template 38 x 42','intensity'),(course_r.values,'Valid 219 x 215; GT=(218,214)','intensity²')],title='Exact minimum FFT shape 293 x 297; no course-image reuse')
    fig,axes=plt.subplots(1,2,figsize=(15,5));names=list(scenes);pos=np.arange(len(names));width=.19
    for j,method in enumerate(METHODS):
        rows=[next(r for r in measurements if r['scene']==n and r['method']==method) for n in names]
        axes[0].bar(pos+(j-1.5)*width,[np.nan if r['error_pixels'] is None else r['error_pixels'] for r in rows],width,label=method)
    axes[0].set_xticks(pos,names,rotation=55,ha='right');axes[0].set(ylabel='Localization error [pixel]',title='Absent case has no GT error; see false-positive trials');axes[0].legend(fontsize=8)
    for category,color in [('background','tab:blue'),('near_distractor','tab:red')]:
        rows=[r for r in trials if r['category']==category];axes[1].scatter([r['seed'] for r in rows],[r['score'] for r in rows],label=category,color=color)
    axes[1].axhline(.8,color='black',linestyle='--',label='illustrative threshold .8');axes[1].set(xlabel='Background seed',ylabel='Highest ZNCC',title='All trials lack the exact template; near-lookalikes can pass');axes[1].legend();fig.tight_layout();figure_io.save_figure(fig,out/'quantitative.png');plt.close(fig)
    rows=benchmark()
    for name,data in [('measurements',measurements),('sweeps',sweeps),('detection_trials',trials),('benchmark',rows)]:table(out/(name+'.csv'),data)
    info=dict(settings=dict(seed=808,methods=METHODS,tie_atol=1e-12,tie_rtol=float(4096*np.finfo(float).eps),psr_relative_std_floor=float(np.sqrt(np.finfo(float).eps))),scenes=specs,measurements=measurements,sweep_specs=sweep_specs,sweeps=sweeps,detection_trials=trials,course=course_info,
              array_inventory={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
              localization_false_match_rates={m:float(np.mean([r['wrong_location'] for r in measurements if r['method']==m and r['wrong_location'] is not None])) for m in METHODS},
              absent_false_positive_rates={g:float(np.mean([r['accepted'] for r in trials if r['category']==g])) for g in ['background','near_distractor']})
    portable.write_json(out/'metrics.json',info);np.savez_compressed(out/'reference_arrays.npz',**arrays);portable.publish_manifest(out)
    print('Generated 12 figures;',len(arrays),'arrays;',len(measurements),'scene measurements;',len(sweeps),'sweep records; 20 absent trials; 12 runtime records.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'output');run(p.parse_args().output_dir)
