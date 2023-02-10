import pickle

import os, sys
WORKING_DIR = os.path.dirname(os.path.abspath(''))
sys.path.insert(1, os.path.join(WORKING_DIR,'LensQuEst'))


##### 
import warnings
warnings.filterwarnings("ignore")
#####

from universe import *
from halo_fit import *
from cmb import *
from flat_map import *
from weight import *
from pn_2d import *

#####
N_runs = 400
mask_file = 'mask_simple400x400.png'
psfile = 'point_sources.png'
psapod = 1
template_name = mask_file.split('/')[-1].split('.')[0]
template_fname = '%s_point_sources.pkl'%(template_name)
process = True
# number of pixels for the flat map
nX = 400
nY = 400

print(template_fname)
#####

print("Map properties")


mean_field = None

# map dimensions in degrees
sizeX = 10.
sizeY = 10.

# basic map object
baseMap = FlatMap(nX=nX, nY=nY, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)

# multipoles to include in the lensing reconstruction
lMin = 30.; lMax = 3.5e3

# ell bins for power spectra
nBins = 21  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra

print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = StageIVCMB(beam=1., noise=1., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)

# Total power spectrum, for the lens reconstruction
# basiscally gets what we theoretically expect the
# power spectrum will look like
forCtotal = lambda l: cmb.flensedTT(l) + cmb.fdetectorNoise(l)

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

#https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
from scipy.ndimage import gaussian_filter 
from scipy.fft import fft2

mask = rgb2gray(plt.imread('mask_simple%dx%d.png'%(nX, nY)))
apodized_mask = gaussian_filter(mask, 3)
point_sources = rgb2gray(plt.imread(psfile))
point_sources = gaussian_filter(point_sources, psapod) 
apodized_mask += point_sources
nPos = np.where(apodized_mask>1)
apodized_mask[nPos] = 1
mask = 1-mask
apodized_mask = 1 - apodized_mask

for a in apodized_mask:
    for b in a:
        assert(b<=1 and b>=0)



from tqdm import trange 

for i in trange(N_runs):
#    print('Run %d of %d'%(i, N_runs))

#    print("\tGenerate GRF unlensed CMB map (debeamed)")
    cmb0Fourier = baseMap.genGRF(cmb.funlensedTT, test=False)
    cmb0 = baseMap.inverseFourier(cmb0Fourier)

#    print("\tGenerate GRF kappa map")
    kCmbFourier = baseMap.genGRF(p2d_cmblens.fPinterp, test=False)
    kCmb = baseMap.inverseFourier(kCmbFourier)


#    print("\tLens the CMB map")
    lensedCmb = baseMap.doLensing(cmb0, kappaFourier=kCmbFourier)
    lensedCmbFourier = baseMap.fourier(lensedCmb)


#    print("\tGenerate FG map")
    fgFourier = baseMap.genGRF(cmb.fForeground, test=False)
    lensedCmbFourier = lensedCmbFourier + fgFourier
    lensedCmb = baseMap.inverseFourier(lensedCmbFourier)


#    print("\tAdd white detector noise (debeamed)")
    noiseFourier = baseMap.genGRF(cmb.fdetectorNoise, test=False)
    totalCmbFourier = lensedCmbFourier + noiseFourier
    totalCmb = baseMap.inverseFourier(totalCmbFourier)


#    print("\tMasking the map")
    totalMaskedCmb = apodized_mask*totalCmb
    totalMaskedCmbFourier = baseMap.fourier(totalMaskedCmb)


    kappa_map = baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalMaskedCmbFourier)
    if(mean_field is None):
        mean_field = kappa_map
    else:
        mean_field += kappa_map
    f = open(template_fname, 'wb') 
    pickle.dump(mean_field/(i+1), f)


f = open(template_fname, 'wb') 
pickle.dump(mean_field/N_runs, f)

print(mean_field/N_runs)
