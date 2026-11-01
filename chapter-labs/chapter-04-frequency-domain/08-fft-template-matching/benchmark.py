"""Small environment-specific measurements; no runtime pass/fail assertions."""
import time,tracemalloc
import numpy as np
from template_matching import match_template
from matching_scenes import geometric_template
from matching_reference import maps


def measure(repeats=3):
    rows=[]
    for H,W,h,w in [(32,48,5,7),(64,80,9,11),(128,128,15,17)]:
        image=.2+np.random.default_rng(8080+H).normal(0,.04,(H,W));template=geometric_template((h,w))
        image[H//3:H//3+h,W//3:W//3+w]=template
        references,_=maps(image,template)
        for method in ['direct','fft','zncc','ssd']:
            match_template(image,template,method=method)
            samples=[]
            for _ in range(repeats):
                start=time.perf_counter();result=match_template(image,template,method=method);samples.append(time.perf_counter()-start)
            tracemalloc.start();match_template(image,template,method=method);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            rows.append(dict(image_rows=H,image_columns=W,template_rows=h,template_columns=w,method=method,repeats=repeats,
                             median_seconds=float(np.median(samples)),tracemalloc_peak_bytes=int(peak),
                             max_absolute_error=float(np.max(abs(result.response-references[method])))))
    return rows
