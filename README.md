#  CMB lensing: standard quadratic estimator, shear and magnification [1804.06403](https://arxiv.org/abs/1804.06403).
## Fork for "Auto from cross: CMB lensing power spectrum without noise bias" [2402.04309](https://arxiv.org/abs/2402.04309)


Manipulate flat sky maps (FFT, filtering, power spectrum, generating Gaussian random field, applying lensing to a map, etc).
Forecast the noise on CMB lensing estimators (standard, shear-only, magnification-only).
Evaluate these estimators on flat sky maps.

To get setup with required packages
```
conda env create -f environment.yml
conda activate nblensing
python -m ipykernel install --user --name nblensing --display-name "nblensing"
```
Or you can take a look into `environment.yml` to get what you need

Demo in `demos/demo.ipynb`


For [2402.04309](https://arxiv.org/abs/2402.04309), the FFT computation of our proposed estimator can be found in `LensQuEst/flat_map.py` under the function `computeQuadEstKappaAutoCorrectionMap` and a demo using this method can be found at the end of `demos/demo.ipynb`. Our numerical experiments are scattered in the `dev/` folder. It is a mess. If you're interested in any of the numerical studies in particular and find the `dev/` folder inpenetrable, please don't hesitate to reach me at delon@stanford.edu!


Hope you find this code useful! if you use this code in a publication, please cite [1804.06403](https://arxiv.org/abs/1804.06403) (+[2402.04309](https://arxiv.org/abs/2402.04309) if relevant). Do not hesitate to contact me with any questions: eschaan@stanford.edu
