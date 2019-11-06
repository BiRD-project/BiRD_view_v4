import json
import numpy as np
import colour as clr

def process_BRDF_json(dict_from_json):
    data = dict_from_json
    wls = np.array(data['data']['wavelengths'])
    bulk_data = np.array(data['data']['data'])

    theta_Vs = np.array(data['data']['theta_v'])
    phi_Vs = np.array(data['data']['phi_v'])
    theta_Is = np.array(data['data']['theta_i'])
    phi_Is = np.array(data['data']['phi_i'])
    pols = np.array(list(data['data']['pol']))

    u_wls = np.unique(wls)
    u_theta_Vs = np.unique(theta_Vs)
    u_phi_Vs = np.unique(phi_Vs)
    u_theta_Is = np.unique(theta_Is)
    u_phi_Is = np.unique(phi_Is)
    u_pols = np.unique(pols)

    pol_row = []
    for pol in u_pols:
        mask_pol = pols == pol
        data_pol = bulk_data[:,mask_pol]
        theta_Is_pol = theta_Is[mask_pol]
        phi_Is_pol = phi_Is[mask_pol]
        theta_Vs_pol = theta_Vs[mask_pol]
        phi_Vs_pol = phi_Vs[mask_pol]
        thetaI_row = []
        for theta_I in u_theta_Is:
            mask_pol_thetaI = theta_Is_pol == theta_I
            data_pol_thetaI = data_pol[:,mask_pol_thetaI]
            phi_Is_pol_thetaI = phi_Is_pol[mask_pol_thetaI]
            theta_Vs_pol_thetaI = theta_Vs_pol[mask_pol_thetaI]
            phi_Vs_pol_thetaI = phi_Vs_pol[mask_pol_thetaI]
            phiI_row = []
            for phi_I in u_phi_Is:
                mask_pol_thetaI_phiI = phi_Is_pol_thetaI == phi_I
                data_pol_thetaI_phiI = data_pol_thetaI[:,mask_pol_thetaI_phiI]
                theta_Vs_pol_thetaI_phiI = theta_Vs_pol_thetaI[mask_pol_thetaI_phiI]
                phi_Vs_pol_thetaI_phiI = phi_Vs_pol_thetaI[mask_pol_thetaI_phiI]
                wl_row = []
                for wl in u_wls:
                    mask_wl = wls == wl
                    data_pol_thetaI_phiI_wl = data_pol_thetaI_phiI[mask_wl,:]
                    thetaV_row = []
                    for theta_V in u_theta_Vs:
                        mask_pol_thetaI_phiI_wl_thetaV = theta_Vs_pol_thetaI_phiI == theta_V
                        data_pol_thetaI_phiI_wl_thetaV = data_pol_thetaI_phiI_wl[:,mask_pol_thetaI_phiI_wl_thetaV]
                        phi_Vs_pol_thetaI_phiI_wl_thetaV = phi_Vs_pol_thetaI_phiI[mask_pol_thetaI_phiI_wl_thetaV]
                        phiV_row = []
                        for phi_v in u_phi_Vs:
                            mask_pol_thetaI_phiI_wl_thetaV_phiV = phi_Vs_pol_thetaI_phiI_wl_thetaV == phi_v
                            data_pol_thetaI_phiI_wl_thetaV_phiV = data_pol_thetaI_phiI_wl_thetaV[:,mask_pol_thetaI_phiI_wl_thetaV_phiV]
                            phiV_row.append(data_pol_thetaI_phiI_wl_thetaV_phiV[0,0])
                        thetaV_row.append(phiV_row)
                    wl_row.append(thetaV_row)
                phiI_row.append(wl_row)
            thetaI_row.append(phiI_row)
        pol_row.append(thetaI_row)
    arranged_data = np.array(pol_row)

    if u_pols.shape[0] > 1:
        arranged_data = np.append(arranged_data, [np.mean(arranged_data, axis=0)], axis=0)
        u_pols = np.append(u_pols, 'average')

    processed_data = {'Wavelengths':u_wls, 'thetaIs':u_theta_Is, 'phiIs':u_phi_Is, 'thetaVs':u_theta_Vs, 'phiVs':u_phi_Vs, 'Polarization':u_pols, 'data':arranged_data}

    return processed_data

# f = open('test_data.json', 'r')
#
# data = json.load(f)
# bulk_data = np.array(data['data']['data'])
# # print(bulk_data[:,0])
#
# u_wls, u_theta_Is, u_phi_Is, u_theta_Vs, u_phi_Vs, u_pols, arranged_data = process_BRDF_json(data)
#
# # a = arranged_data.tolist()
# # print(a)
# pol = 'average'
# phi_i = 0
# theta_i = 0
# wl = 780
# theta_v = -13
# phi_v = 0
#
# selected_data = arranged_data[u_pols == pol, u_theta_Is == theta_i, u_phi_Is == phi_i, u_wls == wl, :, :]
# print(selected_data[0])




