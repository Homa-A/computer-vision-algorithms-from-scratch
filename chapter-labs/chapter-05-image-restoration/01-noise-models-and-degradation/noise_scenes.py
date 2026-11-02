"""Synthetic images and explicit experiment parameters, no third-party assets."""
import numpy as np

SHAPE = (96,128)
SEED = 501
VARIANCE = .04**2
PARAMETERS = {
    'gaussian': (0., .04),
    'rayleigh': (0., 4*VARIANCE/(4-np.pi)),
    'erlang': (np.sqrt(3/VARIANCE), 3.),
    'exponential': (1/np.sqrt(VARIANCE), 1.),
    'uniform': (-np.sqrt(3*VARIANCE), np.sqrt(3*VARIANCE)),
}
PSF = np.array([[0.,1.,2.],[3.,7.,1.],[2.,0.,0.]])/16


def clean_image(shape=SHAPE):
    y, x = np.indices(shape, dtype=float)
    yn, xn = y/max(1,shape[0]-1), x/max(1,shape[1]-1)
    image = .15+.18*xn+.08*yn
    image[(xn>.12)&(xn<.36)&(yn>.2)&(yn<.65)] += .28
    image[(xn-.7)**2+(yn-.38)**2<.12**2] += .3
    texture = (xn>.5)&(yn>.65)
    image[texture] += .025*np.sin(2*np.pi*.16*x[texture])
    image[8:28,8:28] = .28
    return image.astype(float)
