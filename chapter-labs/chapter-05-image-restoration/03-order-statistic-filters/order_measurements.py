"""Measurements from full-precision arrays, no display values."""
import numpy as np


def quality(clean,noisy,out,events):
    e=out-clean;sq=e**2;mse=float(sq.mean())
    avg=lambda a,m:float(a[m].mean()) if m.any() else None
    border=np.zeros(clean.shape,bool);border[:2]=True;border[-2:]=True;border[:,:2]=True;border[:,-2:]=True
    return dict(mse=mse,rmse=float(np.sqrt(mse)),mae=float(abs(e).mean()),psnr_db=float(-10*np.log10(mse)) if mse else None,
        bias=float(e.mean()),variance=0. if np.ptp(e)==0 else float(e.var()),corrupted_mse=avg(sq,events),uncorrupted_mse=avg(sq,~events),
        correction_rate=avg((abs(e)<=.05).astype(float),events),false_alteration_rate=avg((abs(out-noisy)>1e-12).astype(float),~events),
        border_mse=avg(sq,border),interior_mse=avg(sq,~border),gradient_rmse=float(np.sqrt(np.mean((np.diff(out,axis=1)-np.diff(clean,axis=1))**2))))


def details(clean,out,masks):
    result={k+'_survival':float((abs(out[m]-clean[m])<=.1).mean()) for k,m in masks.items()}
    v=out[masks['texture']];truth=clean[masks['texture']]
    result.update(texture_sd_ratio=0. if np.ptp(v)==0 else float(v.std()/truth.std()),profile_rmse=float(np.sqrt(np.mean((out[19]-clean[19])**2))),profile_max_gradient=float(max(abs(np.diff(out[19])))))
    return result
