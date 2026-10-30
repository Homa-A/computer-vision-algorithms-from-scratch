"""Deterministic sharpening experiments; no external images or timing noise."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from shared import lp,png
import sharpen as sh

HERE=Path(__file__).resolve().parent
FIGURES=['transfer.png','radial.png','cutoffs.png','orders.png','methods.png','strength.png','boost_emphasis.png','laplacian.png','profiles.png','boundary.png','quantitative.png']

def write_json(path,value):
 """Force LF on regeneration too, including native Windows Python."""
 Path(path).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8",newline="\n")

def inputs():
 y,x=np.indices((96,128),dtype=float);y/=96;x/=128
 rect=((x>.25)&(x<.75)&(y>.25)&(y<.75)).astype(float)
 clean=.2+.5*rect+.1*np.cos(2*np.pi*4*x)+.06*np.sin(2*np.pi*(18*x+4*y))
 noise=np.random.default_rng(505).normal(0,.04,clean.shape)
 impulse=np.zeros_like(clean);impulse[48,64]=1
 return dict(clean=clean,noisy=clean+noise,noise=noise,constant=np.full(clean.shape,.7),impulse=impulse,step=(x>=.5).astype(float),rectangle=rect,ramp=x.copy(),sinusoid=np.cos(2*np.pi*12*x),texture=np.sin(2*np.pi*(18*x+4*y)))

def generate(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 source=inputs();arrays=dict(source);checks=[];measurements=[];sweep=[];ringing=[]
 def check(name,actual,expected):
  np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
  checks.append(dict(case=name,max_absolute_error=float(abs(np.asarray(actual)-np.asarray(expected)).max())))
 def save(fig,name,title):
  fig.suptitle(title);fig.tight_layout(rect=(0,0,1,.94));png.save_figure(fig,output/name);plt.close(fig)
 def tiles(name,panels,title,cols=3):
  fig,axes=plt.subplots(int(np.ceil(len(panels)/cols)),cols,figsize=(4.2*cols,3.4*int(np.ceil(len(panels)/cols))),squeeze=False)
  for ax,(label,value,signed) in zip(axes.flat,panels):
   value=np.real(value);bound=max(float(abs(value).max()),1e-15)
   im=ax.imshow(value,cmap='RdBu_r' if signed else 'viridis',aspect='auto',interpolation='nearest',vmin=-bound if signed else None,vmax=bound if signed else None)
   ax.set(title=label,xlabel='column (pixel)',ylabel='row (pixel)');fig.colorbar(im,ax=ax,shrink=.8,label='amplitude')
  for ax in axes.flat[len(panels):]:ax.axis('off')
  save(fig,name,title)
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,kind in zip(axes,lp.KINDS):
  mask=sh.high_pass((97,129),.08,kind,centered=True);fy,fx=lp.frequency_axes(mask.shape,centered=True)
  im=ax.imshow(mask,extent=(fx[0],fx[-1],fy[-1],fy[0]),vmin=0,vmax=1,aspect='auto')
  ax.set(title=kind+'; native D₀=.08; n=2',xlabel='column frequency (cycles/pixel)',ylabel='row frequency (cycles/pixel)');fig.colorbar(im,ax=ax,label='amplitude gain')
 save(fig,FIGURES[0],'High-pass complements, odd rectangular grid')
 fig,ax=plt.subplots(figsize=(10,4.5));radial=[]
 cases=[('ideal',2),('butterworth',1),('butterworth',2),('butterworth',4),('butterworth',8),('gaussian',2)]
 for kind,n in cases:
  radius=np.linspace(0,.5,501);gain=1-lp.radial_response(radius,.08,kind,order=n);ax.plot(radius,gain,label=f'{kind} n={n}')
  radial.extend(dict(kind=kind,order=n,radius=float(r),gain=float(g)) for r,g in zip(radius,gain))
 ax.set(xlabel='radial frequency (cycles/pixel)',ylabel='amplitude gain');ax.legend();ax.grid(alpha=.3)
 save(fig,FIGURES[1],'Same native cutoff does not match effective bandwidth')
 panels=[]
 for cutoff in [.04,.08,.16]:
  for kind in lp.KINDS:
   result=sh.apply(source['noisy'],cutoff,kind);panels.append((f'{kind}, D₀={cutoff}',result.values,True))
   sweep.append(dict(kind=kind,cutoff=cutoff,rms=float(np.sqrt(np.mean(abs(result.values)**2)))))
 tiles(FIGURES[2],panels,'Native cutoff effects; signed pure edge fields')
 tiles(FIGURES[3],[(f'Butterworth n={n}',sh.apply(source['noisy'],.08,'butterworth',order=n).values,True) for n in [1,2,4,8]],'Order comparison, native .08 cycles/pixel',2)
 settings=[('hp',{}),('unsharp',{'k':1}),('boost',{'A':2}),('emphasis',{'a':.7,'b':1.4}),('laplacian',{'s':.2})]
 tiles(FIGURES[4],[('Noisy source',source['noisy'],False)]+[(m,sh.apply(source['noisy'],method=m,**kw).values,m=='hp') for m,kw in settings],'Pure edge extraction versus reconstructed sharpening')
 tiles(FIGURES[5],[(f'Unsharp k={k}',sh.apply(source['noisy'],method='unsharp',k=k).values,False) for k in [0,.5,1,2]],'DC-preserving strength and noise',2)
 studies=[('unsharp',{'k':k}) for k in [0,.5,1,2]]+[('boost',{'A':A}) for A in [1,2,3]]+[('emphasis',{'a':a,'b':b}) for a,b in [(1,1),(.7,1.4),(1,2)]]
 tiles(FIGURES[6],[(m+' '+str(kw),sh.apply(source['noisy'],method=m,**kw).values,False) for m,kw in studies[4:]],'High boost changes DC; emphasis has separate DC and detail gains')
 spatial=sh.spatial_laplacian(source['clean']);spectral=lp.apply_transfer(source['clean'],sh.laplacian_response(source['clean'].shape))
 continuous=lp.apply_transfer(source['clean'],sh.laplacian_response(source['clean'].shape,continuous=True))
 check('laplacian_spatial_spectral',spectral,spatial);arrays.update(laplacian_spatial=spatial,laplacian_spectral=spectral)
 tiles(FIGURES[7],[('Independent scalar stencil',spatial,True),('Discrete spectral',spectral,True),('Absolute complex error',abs(spatial-spectral),False),('Continuous spectral comparison',continuous,True)],'Exact finite differences versus a different continuous operator',2)
 fig,ax=plt.subplots(figsize=(11,4.5));positions=np.arange(128)-63.5;ax.plot(positions,source['step'][48],'k--',label='Original step')
 for kind,n in cases:
  edge=sh.apply(source['step'],.08,kind,method='unsharp',order=n);profile=edge.values.real[48];central=profile[40:88]
  arrays['profile_'+kind+str(n)]=profile[None,:]
  ringing.append(dict(kind=kind,order=n,overshoot=float(max(0,central.max()-1)),undershoot=float(max(0,-central.min())),imaginary_residual=edge.imaginary_residual))
  ax.plot(positions,profile,label=f'{kind} n={n}')
 ax.set(xlim=(-32,32),xlabel='offset from edge (pixel)',ylabel='amplitude');ax.grid(alpha=.3);ax.legend()
 save(fig,FIGURES[8],'Unit-step USM, k=1; immediate overshoot and oscillatory tails distinguished')
 boundary=np.zeros((64,96));boundary[16:48,:10]=1;arrays['boundary_input']=boundary;panels=[('Source',boundary,False)]
 for i,(padding,mode) in enumerate([((0,0),'zero'),((32,32),'zero'),((32,32),'reflect')]):
  r=sh.apply(boundary,method='unsharp',padding=padding,boundary=mode);arrays['boundary_'+str(i)]=r.values;panels.append((f'{mode}, pad={padding}',r.values,False))
 tiles(FIGURES[9],panels,'Explicit extension, fixed physical cutoff, original-size crop',2)
 parameter_rows=[]
 for m,kw in studies:
  r=sh.apply(source['noise'],method=m,**kw);edge=sh.apply(source['step'],method=m,**kw).values.real[48,40:88]
  dc=1 if m=='unsharp' else kw['A']-1 if m=='boost' else kw['a']
  parameter_rows.append(dict(method=m,parameters=json.dumps(kw,sort_keys=True),dc=dc,noise_power_gain=float(np.mean(abs(r.values)**2)/np.mean(source['noise']**2)),overshoot=float(max(0,edge.max()-dc))))
 fig,axes=plt.subplots(1,2,figsize=(13,5));labels=[r['method']+' '+r['parameters'] for r in parameter_rows]
 for ax,key in zip(axes,['noise_power_gain','overshoot']):
  ax.bar(np.arange(10),[r[key] for r in parameter_rows]);ax.set_xticks(np.arange(10),labels,rotation=70,ha='right');ax.set_ylabel('power ratio' if key=='noise_power_gain' else 'amplitude');ax.set_title(key);ax.grid(axis='y',alpha=.3)
 save(fig,FIGURES[10],'Noise amplification and edge overshoot, full precision')
 for m,kw in settings:
  r=sh.apply(source['noisy'],method=m,**kw);clean=sh.apply(source['clean'],method=m,**kw);noise=sh.apply(source['noise'],method=m,**kw)
  check('noise_linearity_'+m,r.values-clean.values,noise.values)
  dc={'hp':0,'unsharp':1,'boost':1,'emphasis':.7,'laplacian':1}[m]
  arrays['filtered_'+m]=r.values
  measurements.append(dict(method=m,dc=dc,mean_error=float(abs(r.values.mean()-dc*source['noisy'].mean())),noise_power_gain=float(np.mean(abs(noise.values)**2)/np.mean(source['noise']**2)),mse=float(np.mean(abs(r.values-source['clean'])**2)),gradient_energy=float(sum(np.mean(abs(np.roll(r.values,-1,i)-r.values)**2) for i in [0,1])),imaginary_residual=r.imaginary_residual))
 for kind in lp.KINDS:
  check('constant_'+kind,sh.apply(source['constant'],.08,kind).values,0)
  check('complement_'+kind,sh.apply(source['clean'],.08,kind).values+lp.filter_image(source['clean'],.08,kind).values,source['clean'])
 arrays['ramp_hp']=sh.apply(source['ramp']).values;arrays['impulse_hp']=sh.apply(source['impulse']).values
 tables=dict(measurements=measurements,parameters=parameter_rows,ringing=ringing,cutoff_sweep=sweep,radial=radial)
 for name,rows in tables.items():
  with (output/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 np.savez_compressed(output/'reference_arrays.npz',**arrays)
 metrics=dict(metadata=dict(python=platform.python_version(),numpy=np.__version__,seed=505,noise_sigma=.04,units='cycles/pixel',normalization='backward',shape=[96,128],figures=FIGURES,array_keys=sorted(arrays)),checks=checks,tables=tables,continuous_discrete_difference=float(abs(continuous-spectral).max()))
 write_json(output/'metrics.json',metrics)
 names=FIGURES+[k+'.csv' for k in tables]+['reference_arrays.npz','metrics.json']
 manifest=[dict(path=n,bytes=(output/n).stat().st_size,sha256=hashlib.sha256((output/n).read_bytes()).hexdigest()) for n in names]
 write_json(output/'artifact_manifest.json',manifest)
 for n in FIGURES:png.verify_png(output/n)
 print(f'Generated {len(FIGURES)} verified figures, {len(arrays)} arrays, {len(checks)} exact checks.')
 print('Maximum error:',max(r['max_absolute_error'] for r in checks))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=HERE/'output');args=p.parse_args();generate(args.output_dir)
