"""Original deterministic localization scenes and separate source/observation."""
import numpy as np


def geometric_template(shape=(9,11)):
    y,x=np.indices(shape);h,w=shape
    t=np.full(shape,.12)
    t[(x>=w//5)&(x<=w//2)&(y>=1)&(y<h-1)]+=.55
    t[(y>=h//2)&(y<=h//2+1)&(x>=w//3)&(x<w-1)]+=.3
    t[(x==w-2)&(y==1)]+=.75
    return t


def make_scenes():
    scenes={};template=geometric_template()
    def build(name,shape=(48,64),position=(18,27),t=None,seed=808,transform=None,sigma=0.,positions=None):
        t=template.copy() if t is None else t.copy()
        rng=np.random.default_rng(seed);image=.08+rng.normal(0,.02,shape)
        locations=[position] if positions is None else positions
        for y,x in locations:
            obj=t if transform is None else transform(t.copy())
            image[y:y+obj.shape[0],x:x+obj.shape[1]]=obj
        clean=image.copy();observed=clean+np.random.default_rng(seed+1000).normal(0,sigma,shape)
        scenes[name]=dict(clean=clean,image=observed,template=t,locations=[list(p) for p in locations],
                          seed=seed,sigma=sigma,present=bool(locations),description=name)
    build('unique')
    build('top_left',position=(0,0))
    build('bottom_right',position=(39,53))
    build('odd_even',shape=(47,61),position=(29,38),t=geometric_template((8,10)))
    build('repeated',positions=[(4,5),(29,42)])
    build('distractors',position=(27,35))
    for y,x in [(3,5),(14,49)]:
        distortion=template+.17*np.random.default_rng(y+x).normal(size=template.shape)
        scenes['distractors']['image'][y:y+9,x:x+11]=distortion
    scenes['distractors']['clean']=scenes['distractors']['image'].copy()
    build('affine',transform=lambda t:1.8*t+.4)
    build('brightness_decoy',position=(5,7),transform=lambda t:1.8*t+.4)
    y,x=np.indices(template.shape)
    scene=scenes['brightness_decoy'];scene['image'][25:34,8:19]=template+.09*np.cos(.8*x+1.1*y)
    scene['image'][12:21,43:54]=3.
    scene['clean']=scene['image'].copy()
    build('noise',sigma=.16)
    def occlude(t):t[:,t.shape[1]//2:]=.08;return t
    build('occlusion',transform=occlude)
    build('rotated',position=(20,30),transform=lambda t:np.rot90(t))
    build('scaled',position=(20,28),transform=lambda t:np.repeat(np.repeat(t,2,0),2,1))
    build('near_constant',t=2+1e-10*(template-template.mean()))
    build('low_texture');s=scenes['low_texture'];s['image'][:]=.08;s['image'][18:27,27:38]=template;s['clean']=s['image'].copy()
    build('absent',positions=[],seed=899)
    return scenes


def sweep_scene(kind,value):
    scene=make_scenes()['unique'];image=scene['clean'].copy();t=scene['template'];y,x=scene['locations'][0]
    if kind=='noise':image+=np.random.default_rng(1808).normal(0,value,image.shape)
    elif kind=='occlusion':
        count=int(np.floor(value*t.shape[1]));image[y:y+t.shape[0],x+t.shape[1]-count:x+t.shape[1]]=.08
    elif kind=='contrast':image[y:y+t.shape[0],x:x+t.shape[1]]=value*t+.3
    else:raise ValueError('Unknown sweep')
    return image,t,[y,x]
