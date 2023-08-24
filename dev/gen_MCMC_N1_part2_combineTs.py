import pickle
import warnings
import sys

DATA_IDX_1 = eval(sys.argv[1])
DATA_IDX_2 = eval(sys.argv[2])



import os, sys
WORKING_DIR = os.path.dirname(os.path.abspath(''))
sys.path.insert(1, os.path.join(WORKING_DIR,'LensQuEst'))
from tqdm import tqdm,trange

from universe import *
from halo_fit import *
from cmb import *
from flat_map import *
from weight import *
from pn_2d import *
import pickle
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np
#######
IN_DATA_FNAMES = ['/oak/stanford/orgs/kipac/users/delon/LensQuEst/map_sims_%d.pkl'%(i) for i in range(1,51)]


pairs = [
#   [0,0], #N0
#   [0,1], #kappa
#   [1,0], #kappa
#   [0,2], #N1
#   [1,1], #N1
#   [2,0], #N1
#    [0,3], #should vanish
#    [1,2], #should vanish
#    [2,1], #should vanish
#    [3,0], #should vanish
#    [0,4], #N2 
#    [1,3], #N2
#    [2,2], #N2
#    [3,1], #N2
#    [4,0], #N2
   [-1, -1], #QE
   [-2, -2], #unlensed
]



warnings.filterwarnings("ignore")
#####

oup_fname = '../data/input/universe_Planck15/camb/CAMB_outputs.pkl'
print(oup_fname)
f = open(oup_fname, 'rb') 
powers,cl,c_lensed,c_lens_response = pickle.load(f)
f.close()

totCL=powers['total']
unlensedCL=powers['unlensed_scalar']

L = np.arange(unlensedCL.shape[0])

unlensedTT = unlensedCL[:,0]/(L*(L+1))*2*np.pi
F = unlensedTT
funlensedTT = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

L = np.arange(cl.shape[0])
PP = cl[:,0]
rawPP = PP*2*np.pi/((L*(L+1))**2)
rawKK = L**4/4 * rawPP

fKK = interp1d(L, rawKK, kind='linear', bounds_error=False, fill_value=0.)

L = np.arange(totCL.shape[0])

lensedTT = totCL[:,0]/(L*(L+1))*2*np.pi
F = lensedTT
flensedTT = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)


ftot = lambda l : flensedTT(l) + cmb.fForeground(l) + cmb.fdetectorNoise(l)


L = np.arange(c_lens_response.shape[0])

cTgradT = c_lens_response.T[0]/(L*(L+1))*2*np.pi

fTgradT = interp1d(L, cTgradT, kind='linear', bounds_error=False, fill_value=0.)

# In[3]:



# In[4]:




# In[5]:


print("Map properties")

# number of pixels for the flat map
nX = 1200
nY = 1200

# map dimensions in degrees
sizeX = 20.
sizeY = 20.

# basic map object
baseMap = FlatMap(nX=nX, nY=nY, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)

# multipoles to include in the lensing reconstruction
lMin = 30.; lMax = 3.5e3

# ell bins for power spectra
nBins = 51  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra


# In[6]:


print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = StageIVCMB(beam=1.4, noise=7., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)

# Total power spectrum, for the lens reconstruction
# basiscally gets what we theoretically expect the
# power spectrum will look like
forCtotal = lambda l: ftot(l) 

# # reinterpolate: gain factor 10 in speed
cmb.fCtotal = ftot # no longer reinterpolating since it seems like it leads to errors?


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


# In[2]:




# In[3]:


## kappa_fixed = baseMap.genGRF(fKK, test=False)


TkTkps = []
TkpTks = []
TTps = []
TpTs = []




with open('/oak/stanford/orgs/kipac/users/delon/LensQuEst/N1-mcmc-morestats-Ts%d.pkl'%(DATA_IDX_1), "rb") as f:
    TkTkps, TkpTks, _, _ = pickle.load(f)

with open('/oak/stanford/orgs/kipac/users/delon/LensQuEst/N1-mcmc-morestats-Ts%d.pkl'%(DATA_IDX_2), "rb") as f:
    _, _, TTps, TpTs = pickle.load(f)




def tmp_combine_Cl(Cls_tot):
    n_runs = np.shape(Cls_tot)[0]
    lCen = Cls_tot[0][0]
    Cls = np.sum(np.transpose(Cls_tot, axes=[1,2,0])[1], axis=1)
    sCls = np.sum(np.transpose(Cls_tot, axes=[1,2,0])[2], axis=1)
    return lCen, Cls, sCls


# In[8]:


nBins = 51


# In[9]:


import multiprocessing
c_data = None
ck = 'N1'
def compute_data(i, j):
    TTp = TTps[i]
    TpT = TpTs[i]
    TkTkp = TkTkps[j]
    TkpTk = TkpTks[j]
    curr_data = []

    for s, a, b in [[1, TkTkp, TkTkp], [1, TkTkp, TkpTk], [-1, TTp, TTp], [-1, TTp, TpT]]:
        t0, t1, t2 = baseMap.crossPowerSpectrum(dataFourier1=a, dataFourier2=b, nBins=nBins)
        curr_data.append([t0, s * t1, t2])

    c_ps_data = {}
    c_ps_data[ck] = [0, 0, 0]
    c_ps_data[ck][0], c_ps_data[ck][1], c_ps_data[ck][2] = tmp_combine_Cl(curr_data)
    
    return c_ps_data[ck]

pool = multiprocessing.Pool()

results = []
for i in trange(len(TTps)):
    for j in range(len(TkTkps)):
        results.append(pool.apply_async(compute_data, (i, j)))

pool.close()
pool.join()

for result in tqdm(results):
    if c_data is None:
        c_data = np.array([result.get()])
    else:
        c_data = np.vstack((c_data, np.array([result.get()])))

print(c_data.shape)

with open('/oak/stanford/orgs/kipac/users/delon/LensQuEst/N1-mcmc-nBins%d-morestats-Tcombos_%d_%d.pkl'%(nBins, DATA_IDX_1, DATA_IDX_2), "wb") as f:
    pickle.dump(c_data, f)
