import pickle
import warnings

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
nY =1200

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


# In[6]:


print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = StageIVCMB(beam=1.4, noise=7., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)

# Total power spectrum, for the lens reconstruction
# basiscally gets what we theoretically expect the
# power spectrum will look like
forCtotal = lambda l: ftot(l) 

# reinterpolate: gain factor 10 in speed
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
F = np.array(list(map(forCtotal, L)))
cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)


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

d_idx = eval(sys.argv[1])
d = in_data['totalF_1'][d_idx]

    
# d_idx = eval(sys.argv[1])

# IN_DATA_IDX = d_idx//10
# fname = IN_DATA_FNAMES[IN_DATA_IDX]

# f = open(fname, 'rb') 
# c_in_data = pickle.load(f) 
# f.close()
# for key in c_in_data:
#     if(key not in in_data.keys()):
#         in_data[key] = np.array(c_in_data[key])
#     else:
#         in_data[key] = np.vstack( (in_data[key],np.array(c_in_data[key])) )


# d = in_data['totalF_1'][d_idx%10]

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

    
RDN0_data = {}


ds1s = []
s1ds = []
s1s2s= []
s2s1s= []


tmp_idx = 0
for s_idx in range(50):
    print('CURR', s_idx)
    for s2_idx in trange(50):
        if(s_idx != 49 or s2_idx != 49):
            tmp_idx += 1 
            continue
        s1 = in_data['totalF_0'][s_idx]
        s2 = in_data['totalF_0'][s2_idx+len(in_data['totalF_0'])//2]

        ds1 = baseMap.computeQuadEstKappaNorm(fTgradT, cmb.fCtotal, 
                                             lMin=lMin, lMax=lMax, 
                                             dataFourier=d,
                                             dataFourier2=s1)
        s1d = baseMap.computeQuadEstKappaNorm(fTgradT, cmb.fCtotal, 
                                             lMin=lMin, lMax=lMax, 
                                             dataFourier=s1,
                                             dataFourier2=d)
        s1s2 = baseMap.computeQuadEstKappaNorm(fTgradT, cmb.fCtotal, 
                                             lMin=lMin, lMax=lMax, 
                                             dataFourier=s1,
                                             dataFourier2=s2)
        s2s1 = baseMap.computeQuadEstKappaNorm(fTgradT, cmb.fCtotal, 
                                             lMin=lMin, lMax=lMax, 
                                             dataFourier=s2,
                                             dataFourier2=s1)

#         if(len(ds1s)==0):
#             ds1s = np.array([ds1])
#         else:
#             ds1s = np.vstack((ds1s, np.array([ds1])))

#         if(len(s1ds)==0):
#             s1ds = np.array([s1d])
#         else:
#             s1ds = np.vstack((s1ds, np.array([s1d])))

#         if(len(s1s2s)==0):
#             s1s2s = np.array([s1s2])
#         else:
#             s1s2s = np.vstack((s1s2s, np.array([s1s2])))

#         if(len(s2s1s)==0):
#             s2s1s = np.array([s2s1])
#         else:
#             s2s1s = np.vstack((s2s1s, np.array([s2s1])))
            
            
            
            
        RDN0_data = {
            'ds1' : ds1,
            's1d' : s1d,
            's1s2': s1s2,
            's2s1': s2s1
        }


        oup_fname = '/scratch/users/delon/LensQuEst/RDN0-in_data-%d-%d.pkl'%(d_idx,tmp_idx)
        print(oup_fname)
        f = open(oup_fname, 'wb') 
        pickle.dump(RDN0_data, f)
        f.close()
        tmp_idx += 1
        
        del ds1
        del s1d
        del s1s2
        del s2s1


# RDN0_data = {
#     'ds1s' : ds1s,
#     's1ds' : s1ds,
#     's1s2s': s1s2s,
#     's2s1s': s2s1s
# }


# oup_fname = '/scratch/users/delon/LensQuEst/RDN0-in_data-%d.pkl'%(d_idx)
# print(oup_fname)
# f = open(oup_fname, 'wb') 
# pickle.dump(RDN0_data, f)
# f.close()


