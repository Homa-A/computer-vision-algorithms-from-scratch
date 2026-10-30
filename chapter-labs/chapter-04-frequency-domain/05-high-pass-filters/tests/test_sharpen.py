import numpy as np
import pytest
from numpy.testing import assert_allclose as close
import sharpen as h
from shared import lp

T=dict(rtol=1e-10,atol=1e-10)
SHAPES=[(1,1),(1,7),(6,1),(5,7),(6,8),(7,6)]
KINDS=['ideal','butterworth','gaussian']

@pytest.mark.parametrize('shape',SHAPES)
@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('complex_input',[False,True])
def test_masks_modes_constants(shape,kind,complex_input):
 mask=h.high_pass(shape,.125,kind);assert mask[0,0]==0
 assert np.all((mask>=0)&(mask<=1))
 close(mask+lp.transfer_function(shape,.125,kind),1,**T)
 u,v=np.indices(shape);close(mask[(-u)%shape[0],(-v)%shape[1]],mask,rtol=0,atol=0)
 close(np.fft.ifftshift(h.high_pass(shape,.125,kind,centered=True)),mask,rtol=0,atol=0)
 constant=np.full(shape,2+3j if complex_input else 2.)
 r=h.apply(constant,.125,kind);close(r.values,0,**T)
 assert r.imaginary_residual is None if complex_input else r.imaginary_residual<1e-12
 mode=np.exp(2j*np.pi*(u/shape[0]+v/shape[1]))
 d=np.hypot(0 if shape[0]==1 else 1/shape[0],0 if shape[1]==1 else 1/shape[1])
 gain=float(d>.125) if kind=='ideal' else (d/.125)**4/(1+(d/.125)**4) if kind=='butterworth' else -np.expm1(-d*d/(2*.125**2))
 close(h.apply(mode,.125,kind).values,gain*mode,**T)

@pytest.mark.parametrize('shape',SHAPES)
@pytest.mark.parametrize('method,dc',[('hp',0),('unsharp',1),('boost',2),('emphasis',.7),('laplacian',1)])
def test_operator_identities(shape,method,dc):
 rng=np.random.default_rng(505);x=rng.normal(size=shape)+1j*rng.normal(size=shape)
 low=lp.filter_image(x,.1,'gaussian').values
 expected={'hp':x-low,'unsharp':x+1.4*(x-low),'boost':3*x-low,'emphasis':.7*x+1.2*(x-low),'laplacian':x-.3*h.spatial_laplacian(x)}[method]
 actual=h.apply(x,.1,method=method,k=1.4,A=3,a=.7,b=1.2,s=.3).values
 close(actual,expected,**T);close(actual.mean(),dc*x.mean(),**T)

@pytest.mark.parametrize('shape',SHAPES)
def test_discrete_laplacian(shape):
 x=np.random.default_rng(506).normal(size=shape)
 expected=sum(np.roll(x,delta,axis) for axis in [0,1] for delta in [-1,1])-4*x
 close(h.spatial_laplacian(x),expected,**T)
 close(lp.apply_transfer(x,h.laplacian_response(shape)),expected,**T)
 u,v=np.indices(shape);mode=np.exp(2j*np.pi*(u/shape[0]+v/shape[1]))
 eigen=-4*(np.sin(np.pi/shape[0])**2+np.sin(np.pi/shape[1])**2)
 close(h.spatial_laplacian(mode),eigen*mode,**T)

@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('cutoff',[1e-300,.08,1e300])
@pytest.mark.parametrize('order',[1,2,4,8,1024])
def test_stability(kind,cutoff,order):
 r=h.high_pass((5,8),cutoff,kind,order=order)
 assert r[0,0]==0 and np.isfinite(r).all() and np.all((r>=0)&(r<=1))

@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('boundary',['zero','reflect'])
def test_padding_and_shift(kind,boundary):
 x=np.random.default_rng(507).normal(size=(5,8));padded=np.pad(x,((2,2),(3,3)),mode='constant' if boundary=='zero' else 'reflect')
 fy=np.fft.fftfreq(9)[:,None];fx=np.fft.fftfreq(14)[None,:];d=np.sqrt(fy*fy+fx*fx)
 mask=(d>.1) if kind=='ideal' else (d/.1)**4/(1+(d/.1)**4) if kind=='butterworth' else -np.expm1(-d*d/.02)
 expected=np.fft.ifft2(np.fft.fft2(padded)*mask)
 r=h.apply(x,.1,kind,padding=(2,3),boundary=boundary,centered=True)
 close(r.values,expected[2:7,3:11],**T);assert r.imaginary_residual<1e-12 and r.crop_start==(2,3)


def test_impulse_convolution_and_boundary():
 x=np.arange(35,dtype=float).reshape(5,7);saved=x.copy();kernel=np.fft.ifft2(h.high_pass(x.shape,.12));expected=np.zeros(x.shape,complex)
 for a,b in np.ndindex(x.shape):expected+=kernel[a,b]*np.roll(x,(a,b),(0,1))
 close(h.apply(x,.12).values,expected,**T);close(x,saved,rtol=0,atol=0)
 impulse=np.zeros(x.shape);impulse[0,0]=1;close(h.apply(impulse,.12).values,kernel,**T)
 assert h.high_pass((8,16),.125,'ideal')[0,2]==0
 assert h.high_pass((8,16),.125,'ideal')[0,3]==1
 assert h.laplacian_response((8,16))[4,0]==-4
 close(h.laplacian_response((8,16),continuous=True)[4,0],-np.pi**2,**T)


def test_invalid_and_numeric_capabilities():
 for v in [np.longdouble(.1),np.longdouble(np.finfo(float).max),np.finfo(float).max]:assert np.isfinite(h.high_pass((3,5),v)).all()
 if np.finfo(np.longdouble).max>np.finfo(float).max:
  with pytest.raises(ValueError):h.high_pass((3,5),np.finfo(np.longdouble).max)
 else:assert np.isfinite(h.high_pass((3,5),np.finfo(np.longdouble).max)).all()
 for v in [10**400,0,-1,np.nan,np.inf,True,'x']:
  with pytest.raises((ValueError,TypeError)):h.high_pass((3,5),v)
 for kwargs in [dict(k=-1),dict(A=.5),dict(a=np.inf),dict(b=True),dict(s=-1),dict(method='x'),dict(order=0),dict(padding=(-1,2)),dict(boundary='x'),dict(centered=1)]:
  with pytest.raises(ValueError):h.apply([[1]],**kwargs)
 for x in [[],[1,2],[[np.inf]],[['x']],np.ones((2,3,1))]:
  with pytest.raises((ValueError,TypeError)):h.apply(x)
 for shape in [(0,3),(True,3),(1,),(2.5,3)]:
  with pytest.raises(ValueError):h.high_pass(shape,.1)
