from lensingbiases._lensing_biases import lensingbiases as bias

#first we need ClTT
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
nBins = 51  # number of bins
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

ell = list([i for i in lRange])

lensedcmbfile = "n1_data/lensedCls.dat"
f = open(lensedcmbfile, "w") 
print('# %15s %15s %15s %15s %15s'%('L', 'TT', 'EE', 'BB', 'TE'), file=f)
for l in range(int(lRange[0])+1, int(lRange[1])):
    print('  %15s %15e %15e %15e %15e'%(l,
                                         (l*l+1)/(2*np.pi)*cmb.ftotal(l),
                                        (l*l+1)/(2*np.pi)*cmb.flensedEE(l),
                                        (l*l+1)/(2*np.pi)*cmb.flensedBB(l), 
                                        (l*l+1)/(2*np.pi)*cmb.flensedTE(l)), file=f)
f.close()

phifile = "n1_data/lenspotentialCls.dat"
f = open(phifile, "w") 
print('# %15s %15s %15s %15s %15s %15s %15s %15s'%('L', 'TT', 'EE', 'BB', 'TE', 'PP', 'TP', 'EP'), file=f)


#Taking TP from sample data since it should be good enough 
l_dat = np.loadtxt('n1_data/test_data_set_lenspotentialCls.dat').T[0]
TP = np.loadtxt('n1_data/test_data_set_lenspotentialCls.dat').T[-2]
fTP = interp1d(l_dat, TP, kind='linear', bounds_error=False, fill_value=0.)

for l in range(int(lRange[0])+1, int(lRange[1])):
    KK = p2d_cmblens.fPinterp(l)
    
    #do twice since we're considering power spectrum <phi phi> and <kk>
    #not just k or phi
    phiphi =  -2. * KK / l**2
    phiphi =  -2. * phiphi / l**2
    phiphi = (l*(l+1))**2/(2*np.pi)* phiphi #convention from CAMB

    print('  %15s %15e %15e %15e %15e %15e %15e %15e'%(l,
                                                       (l*l+1)/(2*np.pi)*cmb.funlensedTT(l),
                                                       (l*l+1)/(2*np.pi)*cmb.funlensedEE(l),
                                                       (l*l+1)/(2*np.pi)*cmb.funlensedBB(l),
                                                       (l*l+1)/(2*np.pi)*cmb.funlensedTE(l),
                                                       phiphi,
                                                       fTP(l),0), file=f)
f.close()

FWHM = 1.4 #arcmin
noise_level = 7 #muK*arcmin

lmin = 2

lmax = lMax
lmaxout = lmax
lmax_TT = lmax
tmp_output = 'n1_data'
print('computing N1')

#### TEST WITH SAMPLE DATA
phifile = 'n1_data/test_data_set_lenspotentialCls.dat'
lensedcmbfile = 'n1_data/test_data_set_lensedCls.dat'
FWHM = 3.5 #arcmin
noise_level = 17.72 #muK*arcmin
####

bias.compute_n0(phifile, lensedcmbfile, FWHM/60,
                noise_level,
                lmin,
                lmaxout,
                lmax,
                lmax_TT,
                0,
                tmp_output)

bias.compute_n1(phifile, lensedcmbfile, FWHM/60,
                noise_level,
                lmin,
                lmaxout,
                lmax,
                lmax_TT,
                0,
                tmp_output)