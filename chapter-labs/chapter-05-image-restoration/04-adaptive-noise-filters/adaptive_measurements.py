"""Full precision metrics; no display clipping, null for empty denominators."""
import numpy as np


def quality(clean,noisy,out,events):
    e=out-clean;sq=e*e;mse=float(sq.mean());avg=lambda v,m:float(v[m].mean()) if m.any() else None
    b=np.zeros(clean.shape,bool);b[:2]=True;b[-2:]=True;b[:,:2]=True;b[:,-2:]=True
    return dict(mse=mse,rmse=float(np.sqrt(mse)),mae=float(abs(e).mean()),psnr_db=float(-10*np.log10(mse)) if mse else None,
        bias=float(e.mean()),variance=0. if np.ptp(e)==0 else float(e.var()),corrupted_mse=avg(sq,events),uncorrupted_mse=avg(sq,~events),
        correction_rate=avg((abs(e)<=.05).astype(float),events),false_alteration_rate=avg((abs(out-noisy)>1e-12).astype(float),~events),
        gradient_rmse=float(np.sqrt(np.mean((np.diff(out,axis=1)-np.diff(clean,axis=1))**2))),border_mse=avg(sq,b),interior_mse=avg(sq,~b))


def diagnostic_counts(d):
    unique,count=np.unique(d['size'],return_counts=True)
    return dict(window_histogram={str(int(k)):int(n) for k,n in zip(unique,count)},expansions=int(d['expansions'].sum()),
        stage_a_failures=int(d['stage_a_failures'].sum()),stage_a_failed_pixels=int(d['fallback'].sum()),stage_b_replacements=int(d['stage_b_replaced'].sum()),
        retained=int(d['retained'].sum()),fallbacks=int(d['fallback'].sum()))
