import numpy as np
import json

u_wls = np.arange(450,900,50).tolist()
u_thetaIs = np.arange(0,20,10).tolist()
u_phiIs = np.arange(0,30,10).tolist()
u_thetaVs = np.arange(-85,86,5).tolist()
u_phiVs = np.arange(0,180,30).tolist()
u_pols = ['p','s']

data = []

thetaIs = []
phiIs = []
thetaVs = []
phiVs = []
pols = []

for thetaI in u_thetaIs:
    for phiI in u_phiIs:
        for thetaV in u_thetaVs:
            for phiV in u_phiVs:
                for pol in u_pols:
                    thetaIs.append(thetaI)
                    phiIs.append(phiI)
                    thetaVs.append(thetaV)
                    phiVs.append(phiV)
                    pols.append(pol)

for wl in u_wls:
    b = 0.5+np.sin(np.pi*wl/180)
    spectra = []
    for thetaI in u_thetaIs:
        for phiI in u_phiIs:
            for thetaV in u_thetaVs:
                for phiV in u_phiVs:
                    for pol in u_pols:
                        if pol == 'p':
                            a = 0.5
                        elif pol == 's':
                            a = 1.0
                        else:
                            a = 0.75
                        # value = b*a*np.sin(np.pi*(thetaI+thetaV+phiI+phiV)/180)
                        value = a*b
                        spectra.append(value)
    data.append(spectra)

data = np.array(data)
data = data/np.max(data)
print(data.shape)
data = data.tolist()

example_file = {'head':
                    {'test':
                         'yes'},
                'data':
                    {'wavelengths':u_wls,
                     'theta_i':thetaIs,
                     'phi_i':phiIs,
                     'theta_v':thetaVs,
                     'phi_v':phiVs,
                     'pol':pols,
                     'data':data}
                }


with open('test_data_7.json', 'w') as outfile:
    json.dump(example_file, outfile)
