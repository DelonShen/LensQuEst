import camb
from camb import model, initialpower

#Set up a new set of parameters for CAMB
print('reading in params')
pars = camb.read_ini('../data/input/universe_Planck15/camb/params_planck15_highl.ini')

#calculate results for these parameters
print('computing results')
results = camb.get_results(pars)

#get dictionary of CAMB power spectra
print('getting results')
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
cl = results.get_lens_potential_cls()
c_lensed = results.get_lensed_scalar_cls()
c_lens_response = results.get_lensed_gradient_cls()

print('saving results')
oup_fname = '/oak/stanford/orgs/kipac/users/delon/LensQuEst/CAMB_outputs.pkl'
print(oup_fname)
f = open(oup_fname, 'wb') 
pickle.dump((powers,cl,c_lensed,c_lens_response) , f)
f.close()
