"""Synthetic ground truth; fixed physical frequencies survive shape changes."""
import numpy as np
from notch_support import low_pass


def clean_image(shape=(96,128)):
    shape = low_pass._pair(shape)
    y,x = np.indices(shape,dtype=float)
    # Physical pixel coordinates and reference dimensions are deliberate.
    rectangle = (x>25)&(x<90)&(y>24)&(y<70)
    blob = np.exp(-((x-76)/17)**2-((y-44)/13)**2)
    return (.4+.16*rectangle+.12*blob+
            .06*np.cos(2*np.pi*(18/96)*y)+.035*np.sin(2*np.pi*(5/128*x+3/96*y)))


def periodic_noise(shape, components):
    shape = low_pass._pair(shape)
    y,x = np.indices(shape,dtype=float)
    noise = np.zeros(shape)
    for component in components:
        fy,fx = component['frequency_cpp']
        noise += component['amplitude']*np.cos(2*np.pi*(fy*y+fx*x)+component['phase'])
    return noise


def make_scenes(shape=(96,128)):
    clean = clean_image(shape)
    def part(fy,fx,a=.18,p=.3):
        return dict(frequency_cpp=[fy,fx],amplitude=a,phase=p)
    definitions = dict(
        horizontal=[part(16/96,0)], vertical=[part(0,24/128)],
        diagonal=[part(12/96,20/128)],
        multiple=[part(16/96,0,.14),part(0,24/128,.16,.7),part(9/96,-17/128,.12,1.1)],
        high=[part(0,24/128,.65)], weak=[part(0,24/128,.015)],
        leakage=[part(10.35/96,22.4/128)],
        near_nyquist=[part(0,63/128)], nyquist=[part(0,.5,.18,0.)],
        gaussian_mixed=[part(0,24/128),part(7/96,15/128,.12,.8)])
    scenes = {}
    for name,components in definitions.items():
        periodic = periodic_noise(shape,components)
        gaussian = np.random.default_rng(707).normal(0,.025,shape) if name=='gaussian_mixed' else np.zeros(shape)
        scenes[name]=dict(clean=clean.copy(),periodic=periodic,gaussian=gaussian,
                          noisy=clean+periodic+gaussian,components=components,
                          gaussian_sigma=.025 if name=='gaussian_mixed' else 0.,seed=707)
    return scenes
