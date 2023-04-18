#######
IN_DATA_FNAMES = ['/oak/stanford/orgs/kipac/users/delon/LensQuEst/map_sims_800x800_20x20_%d.pkl'%(i) for i in range(1,11)]
import warnings
warnings.filterwarnings("ignore")
#####


import numpy as np
import os, sys
WORKING_DIR = os.path.dirname(os.path.abspath(''))
sys.path.insert(1, os.path.join(WORKING_DIR,'LensQuEst'))

import sys



#to get latex to work, shoulldn't be necessary for most ppl
os.environ['PATH'] = "%s:/usr/local/cuda-11.2/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/anaconda/bin:/home/delon/texlive/bin/x86_64-linux:/home/delon/.local/bin:/home/delon/bin"%os.environ['PATH']

from universe import *
from halo_fit import *
from cmb import *
from flat_map import *
from weight import *
from pn_2d import *
import pickle
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib
from tqdm import tqdm, trange

print("Map properties")

# number of pixels for the flat map
nX = 800
nY =800

# map dimensions in degrees
sizeX = 20.
sizeY = 20.

# basic map object
baseMap = FlatMap(nX=nX, nY=nY, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)

# multipoles to include in the lensing reconstruction
lMin = 30.; lMax = 3.5e3

# ell bins for power spectra
nBins = 21  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra

#### print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = StageIVCMB(beam=1.4, noise=7., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)

# Total power spectrum, for the lens reconstruction
# basiscally gets what we theoretically expect the
# power spectrum will look like
forCtotal = lambda l: cmb.ftotal(l) 

# reinterpolate: gain factor 10 in speed
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
F = np.array(list(map(forCtotal, L)))
cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

print("CMB lensing power spectrum")
u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)

print("Gets a theoretical prediction for the noise")
fNqCmb_fft = baseMap.forecastN0Kappa(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
Ntheory = lambda l: fNqCmb_fft(l) 

in_data = {}

for fname in tqdm(IN_DATA_FNAMES):
    f = open(fname, 'rb') 
    c_in_data = pickle.load(f) 
    f.close()
    for key in c_in_data:
        if(key not in in_data.keys()):
            in_data[key] = np.array(c_in_data[key])
        else:
            in_data[key] = np.vstack( (in_data[key],np.array(c_in_data[key])) )


for key in in_data:
    print(key, np.shape(in_data[key]))
    
    
pairs = [
    [0,0], #N0
    [0,1], #kappa
    [1,0], #kappa
    [1,1], #N1
    [0,2], #N1
    [2,0], #N1
    [-1, -1], #QE
    [-2, -2], #unlensed
]

data_names = {
    0: 'cmb0F_1',
    1: 'lCmbF_o1_1',
    2: 'lCmbF_o2_1',
    -1: 'lCmbF_1',
    -2: 'totalF_0',
}


ps_data = {}


#RDN0
from tqdm import tqdm,trange

    
d_idx = eval(sys.argv[1])
RDN0_data = {}

d = in_data['totalF_1'][d_idx]

ds1s = []
s1ds = []
s1s2s= []
s2s1s= []

for s_idx in trange(len(in_data['totalF_0'])//2):
#     for s_idx in trange(2):
    s1 = in_data['totalF_0'][s_idx]
    s2 = in_data['totalF_0'][s_idx+len(in_data['totalF_0'])//2]

    ds1 = baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, 
                                         lMin=lMin, lMax=lMax, 
                                         dataFourier=d,
                                         dataFourier2=s1)
    s1d = baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, 
                                         lMin=lMin, lMax=lMax, 
                                         dataFourier=s1,
                                         dataFourier2=d)
    s1s2 = baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, 
                                         lMin=lMin, lMax=lMax, 
                                         dataFourier=s1,
                                         dataFourier2=s2)
    s2s1 = baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, 
                                         lMin=lMin, lMax=lMax, 
                                         dataFourier=s2,
                                         dataFourier2=s1)

    if(len(ds1s)==0):
        ds1s = np.array([ds1])
    else:
        ds1s = np.vstack((ds1s, np.array([ds1])))

    if(len(s1ds)==0):
        s1ds = np.array([s1d])
    else:
        s1ds = np.vstack((s1ds, np.array([s1d])))

    if(len(s1s2s)==0):
        s1s2s = np.array([s1s2])
    else:
        s1s2s = np.vstack((s1s2s, np.array([s1s2])))

    if(len(s2s1s)==0):
        s2s1s = np.array([s2s1])
    else:
        s2s1s = np.vstack((s2s1s, np.array([s2s1])))
RDN0_data = {
    'ds1s' : ds1s,
    's1ds' : s1ds,
    's1s2s': s1s2s,
    's2s1s': s2s1s
}


oup_fname = '/data/delon/LensQuEst/RDN0-in_data-%d.pkl'%(d_idx)
print(oup_fname)
f = open(oup_fname, 'wb') 
pickle.dump(RDN0_data, f)
f.close()
