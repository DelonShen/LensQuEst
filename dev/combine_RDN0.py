#######
IN_DATA_FNAMES = ['/oak/stanford/orgs/kipac/users/delon/LensQuEst/map_sims_800x800_20x20_%d.pkl'%(i) for i in range(1,51)]
import warnings
warnings.filterwarnings("ignore")
#####
import sys
d_idx = eval(sys.argv[1])
nBins = 51
if(len(sys.argv) == 3):
    nBins = eval(sys.argv[2])
import os, sys
WORKING_DIR = os.path.dirname(os.path.abspath(''))
sys.path.insert(1, os.path.join(WORKING_DIR,'LensQuEst'))

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
from tqdm import trange, tqdm

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
lRange = (1., 2.*lMax)  # range for power spectra


# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S4/SO specs
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

# for fname in tqdm(IN_DATA_FNAMES):
#     f = open(fname, 'rb') 
#     c_in_data = pickle.load(f) 
#     f.close()
#     for key in c_in_data:
#         if(key not in in_data.keys()):
#             in_data[key] = np.array(c_in_data[key])
#         else:
#             in_data[key] = np.vstack( (in_data[key],np.array(c_in_data[key])) )


for key in in_data:
    print(key, np.shape(in_data[key]))

    
pairs = [
#    [0,0], #N0
#    [0,1], #kappa
#    [1,0], #kappa
#    [0,2], #N1
#    [1,1], #N1
#    [2,0], #N1
#     [0,3], #should vanish
#     [1,2], #should vanish
#     [2,1], #should vanish
#     [3,0], #should vanish
#     [0,4], #N2 
#     [1,3], #N2
#     [2,2], #N2
#     [3,1], #N2
#     [4,0], #N2
   [-1, -1], #QE
   [-2, -2], #unlensed
]


data_names = {
    0: 'cmb0F_1',
    1: 'lCmbF_o1_1',
    2: 'lCmbF_o2_1',
    3: 'lCmbF_o3_1',
    4: 'lCmbF_o4_1',
    -1: 'lCmbF_1',
    -2: 'totalF_0',
}



data = {}
# # pbar = trange(len(pairs))
# for file_idx in trange(1,51):
#     for pair_idx in range(len(pairs)):
#         pair = pairs[pair_idx]
        
#         f = open('/oak/stanford/orgs/kipac/users/delon/LensQuEst/QE_and_Nhat_from_map_sims_800x800_20x20_Clunlensed_weight_FILE%d_pair_%d_%d.pkl'%(file_idx, pair[0], pair[1]), 'rb')  
#         c_data = pickle.load(f)
#         f.close()        
#         for key in c_data:
#             if(key not in data.keys()):
#                 data[key] = np.array(c_data[key])
#             else:
#                 data[key] = np.vstack((data[key], np.array(c_data[key])))  
# #             print(np.shape(data[key]))



for key in data:
    print(key, np.shape(data[key]))
    
    
def combine_Cl(Cls_tot):
    n_runs = np.shape(Cls_tot)[0]
    print(n_runs, np.shape(Cls_tot))
    lCen = Cls_tot[0][0]
    Cls = np.sum(np.transpose(Cls_tot, axes=[1,2,0])[1], axis=1)/n_runs
    sCls = np.sqrt(np.sum(np.square(np.transpose(Cls_tot, axes=[1,2,0])[2]), axis=1))/n_runs
    return lCen, Cls, sCls

def combine_sketchy(Cl0, Cli):
    n_runs = np.shape(Cl0)[0]
    print(n_runs, np.shape(Cl0))
    ret = np.copy(Cl0)
    ret = np.transpose(ret, axes=[1,2,0])
    ret[1] = np.array([
        [Cl0[run_idx][1][bin_idx]+
         sum([Cli[i][run_idx][1][bin_idx] for i in range(len(Cli))]) 
                     for run_idx in range(n_runs)] 
                    for bin_idx in range(len(Cl0[0][1]))])
    ret[2] = np.array([[np.sqrt(Cl0[run_idx][2][bin_idx]**2+sum([Cli[i][run_idx][2][bin_idx]**2 
                                                               for i in range(len(Cli))]))
                     for run_idx in range(n_runs)] 
                    for bin_idx in range(len(Cl0[0][1]))])
    return np.transpose(ret, axes=[2,0,1])


ps_data = {}

#estimate RDN0
ck = 'RDN(0)'

def tmp_combine_Cl(Cls_tot):
    n_runs = np.shape(Cls_tot)[0]
    lCen = Cls_tot[0][0]
    Cls = np.sum(np.transpose(Cls_tot, axes=[1,2,0])[1], axis=1)
#     sCls =  np.sum(np.transpose(Cls_tot, axes=[1,2,0])[2], axis=1)
    sCls = np.sqrt(np.sum(np.square(np.transpose(Cls_tot, axes=[1,2,0])[2]), axis=1))
    return lCen, Cls, sCls


RDN0_fname = '/oak/stanford/orgs/kipac/users/delon/LensQuEst/RDN0-in_data-%d.pkl'%(d_idx)
RDN0_data = None
try:
    with open(RDN0_fname,"rb") as f:
        RDN0_data = pickle.load(f)
except IOError:
    assert(1==0)
ds1s = RDN0_data['ds1s']
s1ds = RDN0_data['s1ds']
s1s2s= RDN0_data['s1s2s']
s2s1s= RDN0_data['s2s1s']

c_data = None
for i in range(len(ds1s)):
    ds1 = ds1s[i]
    s1d = s1ds[i]
    s1s2= s1s2s[i]
    s2s1= s2s1s[i]
    curr_data = []

    for s,a,b in [[1,ds1,ds1], [1,ds1,s1d], [1,s1d,ds1],[1,s1d,s1d],[-1,s1s2,s1s2],[-1,s1s2,s2s1]]:
        t0, t1, t2 = baseMap.crossPowerSpectrum(dataFourier1=a, 
                                                dataFourier2=b, nBins=nBins)
        curr_data += [[t0,s*t1,t2]] 

    c_ps_data = {}

    c_ps_data[ck] = [0,0,0]

    c_ps_data[ck][0], c_ps_data[ck][1], c_ps_data[ck][2] = tmp_combine_Cl(curr_data)

    if(c_data is None):
        c_data = np.array([c_ps_data[ck]])
    else:
        c_data = np.vstack((c_data, np.array([c_ps_data[ck]])))
        
assert(c_data.shape[0] == len(ds1s))
RDN0_for_data = combine_Cl(c_data)
del c_data
if(ck not in ps_data.keys()):
    ps_data[ck] = np.array([c_ps_data[ck]])
else:
    ps_data[ck] = np.vstack((ps_data[ck], np.array([c_ps_data[ck]])))  

with open('/oak/stanford/orgs/kipac/users/delon/LensQuEst/RDN0-combined-%d-nBins%d.pkl'%(d_idx, nBins), "wb") as f:
    pickle.dump(ps_data, f)
