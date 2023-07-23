
# In[15]:
import sys

#######
DATA_IDX = eval(sys.argv[1])
DATA_FNAME = '/oak/stanford/orgs/kipac/users/delon/LensQuEst/map_sims_%d.pkl'%(DATA_IDX)
print(DATA_FNAME)

N_RUNS = 10
import warnings
warnings.filterwarnings("ignore")
#####


# In[2]:


import os, sys
WORKING_DIR = os.path.dirname(os.path.abspath(''))
sys.path.insert(1, os.path.join(WORKING_DIR,'LensQuEst'))


# In[3]:


from universe import *
from halo_fit import *
from cmb import *
from flat_map import *
from weight import *
from pn_2d import *
import pickle
import seaborn as sns
from scipy.stats import spearmanr


# In[4]:


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


from tqdm import trange,tqdm 
import pickle

from itertools import product

poss = list(product([True, False], range(N_RUNS)))


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

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S4/SO specs
cmb = StageIVCMB(beam=1.4, noise=7., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)

# Total power spectrum, for the lens reconstruction
# basiscally gets what we theoretically expect the
# power spectrum will look like
forCtotal = lambda l: ftot(l) 

# reinterpolate: gain factor 10 in speed
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
F = np.array(list(map(forCtotal, L)))
cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)


f = lambda l: np.sqrt(cmb.fCtotal(l))
clFourier = np.array(list(map(f, baseMap.l.flatten())))
clFourier = np.nan_to_num(clFourier)
clFourier = clFourier.reshape(np.shape(baseMap.l))


data = {}
frandomizePhase = lambda z: np.abs(z) * np.exp(1j*np.random.uniform(0., 2.*np.pi))

for LENSED, run_n in tqdm(poss):
    
    np.random.seed(LENSED*6000000 + DATA_IDX*1483 + run_n )
        
        
    post_fix = '_%d'%(LENSED)

    c_Data = {}
    c_Data['cmb0F'+post_fix] = None
    c_Data['kCmbF'+post_fix] = None
    c_Data['lCmbF'+post_fix] = None
    for i in range(1,5):
        c_Data['lCmbF_o%d'%(i)+post_fix] = None
    c_Data['fgF'+post_fix] = None
    c_Data['noiseF'+post_fix] = None
    c_Data['totalF'+post_fix] = None

    
    totalCmbFourier, totalCmb = None, None
    
    if(not LENSED):
        totalCmbFourier = baseMap.genGRF(ftot)
        c_Data['totalF'+post_fix] = totalCmbFourier
        
        
        dataFourier = np.ones_like(totalCmbFourier)
        dataFourier *= clFourier * np.sqrt(baseMap.sizeX* baseMap.sizeY)

        
        TRand = np.array(list(map(frandomizePhase, dataFourier.flatten())))
        TRand = TRand.reshape(dataFourier.shape)
        
        c_Data['totalF_randomized'+post_fix] = TRand
        
    elif(LENSED):
        cmb0Fourier = baseMap.genGRF(funlensedTT, test=False)
        cmb0 = baseMap.inverseFourier(cmb0Fourier)
        c_Data['cmb0F'+post_fix] = cmb0Fourier
        
        kCmbFourier = baseMap.genGRF(fKK, test=False)
        kCmb = baseMap.inverseFourier(kCmbFourier)
        c_Data['kCmbF'+post_fix] = kCmbFourier
        
#        for i in range(1,5):
#            lensedCmb = baseMap.doLensingTaylor(unlensed=cmb0, kappaFourier=kCmbFourier, order=i)
#            lensedCmbFourier = baseMap.fourier(lensedCmb)
#            c_Data['lCmbF_o%d'%(i)+post_fix] = lensedCmbFourier
            
        lensedCmb = baseMap.doLensing(cmb0, kappaFourier=kCmbFourier)
        lensedCmbFourier = baseMap.fourier(lensedCmb)
        c_Data['lCmbF'+post_fix] = lensedCmbFourier
        
        fgFourier = baseMap.genGRF(cmb.fForeground, test=False)
        c_Data['fgF'+post_fix] = fgFourier
        lensedCmbFourier = lensedCmbFourier + fgFourier

        noiseFourier = baseMap.genGRF(cmb.fdetectorNoise, test=False)
        c_Data['noiseF'+post_fix] = noiseFourier
        totalCmbFourier = lensedCmbFourier + noiseFourier
        
        c_Data['totalF'+post_fix] = totalCmbFourier

    for key in c_Data:
        if(c_Data[key] is None):
            continue
        if(key not in data.keys()):
            data[key] = np.array([c_Data[key]])
        else:
            data[key] = np.vstack((np.array([c_Data[key]]), data[key]))  
            

    f = open(DATA_FNAME, 'wb') 
    p = pickle.Pickler(f) 
    p.fast = True 
    p.dump(data)
    f.close()
