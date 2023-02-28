#######
IN_DATA_FNAME = '/data/delon/LensQuEst/map_sims_800x800_20x20.pkl'
DATA_FNAME = '/data/delon/LensQuEst/QE_and_Nhat_from_map_sims_800x800_20x20.pkl'

preload=False
import warnings
warnings.filterwarnings("ignore")
#####


# In[3]:


import os, sys
WORKING_DIR = os.path.dirname(os.path.abspath(''))
sys.path.insert(1, os.path.join(WORKING_DIR,'LensQuEst'))

#to get latex to work, shoulldn't be necessary for most ppl
os.environ['PATH'] = "%s:/usr/local/cuda-11.2/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/anaconda/bin:/home/delon/texlive/bin/x86_64-linux:/home/delon/.local/bin:/home/delon/bin"%os.environ['PATH']


# In[4]:


from universe import *
from halo_fit import *
from cmb import *
from flat_map import *
from weight import *
from pn_2d import *
import pickle
import seaborn as sns
from scipy.stats import spearmanr


# In[5]:


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


# In[6]:


print("CMB experiment properties")

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


# In[7]:


print("CMB lensing power spectrum")
u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)


# In[8]:


print("Gets a theoretical prediction for the noise")
fNqCmb_fft = baseMap.forecastN0Kappa(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
Ntheory = lambda l: fNqCmb_fft(l) 


# In[9]:


#https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# from scipy.ndimage import gaussian_filter 
# from scipy.fft import fft2

# mask = rgb2gray(plt.imread('mask_simple%dx%d.png'%(nX, nY)))
# apodized_mask = gaussian_filter(mask, 3)
# point_sources = rgb2gray(plt.imread('point_sources_bigger.png'))
# point_sources = gaussian_filter(point_sources, 1.5) 
# apodized_mask += point_sources
# nPos = np.where(apodized_mask>1)
# apodized_mask[nPos] = 1
# mask = 1-mask
# apodized_mask = 1 - apodized_mask

# for a in apodized_mask:
#     for b in a:
#         assert(b<=1 and b>=0)
# plt.imshow(apodized_mask)


# In[12]:


f = open(IN_DATA_FNAME, 'rb') 
in_data = pickle.load(f) 
f.close()
for key in in_data:
    print(key, np.shape(in_data[key]))


# In[30]:


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


# In[ ]:


from tqdm import trange, tqdm

data = {}


# In[43]:


if(preload):
    f = open(DATA_FNAME, 'rb') 
    data = pickle.load(f) 
    f.close()
    for key in data:
        print(key, np.shape(data[key]))

fgFourier = in_data['fgF_1']
noiseFourier = in_data['noiseF_1']

for pair_idx in range(len(pairs)):
    pair = pairs[pair_idx]
    print('currently on %d of %d'%(pair_idx, len(pairs)))
    pair_key = '%d%d'%(pair[0],pair[1])
    keys = [data_names[p] for p in pair]
    print(pair, keys)
    N_data = min(len(in_data[keys[0]]), len(in_data[keys[1]]))
    

    s_idx = 0
    c_data = []
    c_data_sqrtN = []
    c_data_kR  = []

    if(pair_key in data):
        s_idx = len(data[pair_key])
        c_data = data[pair_key]
    
    for data_idx in trange(s_idx, N_data):
        dataF0 = in_data[keys[0]][data_idx]
        if(pair[0]-1 >= 0):  #isolate term
            dataF0 = dataF0 - in_data[data_names[pair[0]-1]][data_idx]
        dataF1 = in_data[keys[1]][data_idx]
        if(pair[1]-1>=0):    #isolate term
            dataF1 = dataF1 - in_data[data_names[pair[1]-1]][data_idx]
        
        if(pair[0]!=-2):
            dataF0 = dataF0 + fgFourier[data_idx] + noiseFourier[data_idx]
            dataF1 = dataF1 + fgFourier[data_idx] + noiseFourier[data_idx]
            
        QE = baseMap.computeQuadEstKappaNorm(cmb.flensedTT, cmb.fCtotal, 
                                             lMin=lMin, lMax=lMax, 
                                             dataFourier=dataF0,
                                             dataFourier2=dataF1)
        sqrtNhat = []
        kR = []
        if(pair[0]==pair[1]):
            sqrtNhat = baseMap.computeQuadEstKappaAutoCorrectionMap(cmb.flensedTT,
                                                                    cmb.fCtotal, 
                                                                    lMin=lMin, lMax=lMax, 
                                                                    dataFourier=dataF0)
            totalCmbFourierRandomized = baseMap.randomizePhases(dataF0)
#             kR = baseMap.computeQuadEstKappaNorm(cmb.flensedTT, cmb.fCtotal, 
#                                                  lMin=lMin, lMax=lMax,
#                                                  dataFourier=totalCmbFourierRandomized)
            if(len(c_data_sqrtN)==0):
                c_data_sqrtN = np.array([sqrtNhat])
            else:
                c_data_sqrtN = np.vstack((c_data_sqrtN, np.array([sqrtNhat])))

#             if(len(c_data_kR)==0):
#                 c_data_kR = np.array([kR])
#             else:
#                 c_data_kR = np.vstack((c_data_kR, np.array([kR])))


        if(len(c_data)==0):
            c_data = np.array([QE])
        else:
            c_data = np.vstack((c_data, np.array([QE])))
            
            
        assert(len(c_data)==data_idx+1)
        
    data[pair_key] = c_data
    data[pair_key+'_sqrtN'] = c_data_sqrtN
#     data[pair_key+'_kR'] = c_data_kR
    f = open(DATA_FNAME, 'wb') 
    pickle.dump(data, f)
    f.close()
