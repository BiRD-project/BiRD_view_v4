import numpy as np
import colour as clr
from colormath.color_objects import sRGBColor

def select_data(Wavelengths, Theta_I, Phi_I, Polarization, Data):
    u_wls = np.array(Data['Wavelengths'])
    u_theta_Is = np.array(Data['thetaIs'])
    u_phi_Is = np.array(Data['phiIs'])
    u_pols = np.array(Data['Polarization'])
    selected_data = np.array(Data['data'])[u_pols == Polarization, u_theta_Is == Theta_I, u_phi_Is == Phi_I, u_wls == Wavelengths, :, :]
    return selected_data[0]

def select_spectrum(Theta_V, Phi_V, Theta_I, Phi_I, Polarization, Data):
    u_theta_Vs = np.array(Data['thetaVs'])
    u_phi_Vs = np.array(Data['phiVs'])
    u_theta_Is = np.array(Data['thetaIs'])
    u_phi_Is = np.array(Data['phiIs'])
    u_pols = np.array(Data['Polarization'])
    selected_spectrum = np.array(Data['data'])[u_pols == Polarization, u_theta_Is == Theta_I, u_phi_Is == Phi_I, :, u_theta_Vs == Theta_V, u_phi_Vs == Phi_V]
    # selected_spectrum = selected_spectrum/np.max(selected_spectrum)
    return selected_spectrum

def get_tristimulus_XYZs(Theta_I, Phi_I, Polarization, Data, observer, illuminant):
    u_wls = np.array(Data['Wavelengths'])
    u_theta_Is = np.array(Data['thetaIs'])
    u_phi_Is = np.array(Data['phiIs'])
    u_pols = np.array(Data['Polarization'])
    u_theta_Vs = np.array(Data['thetaVs'])
    u_phi_Vs = np.array(Data['phiVs'])
    data = np.array(Data['data'])

    cmfs = clr.STANDARD_OBSERVERS_CMFS[observer]
    illuminant = clr.ILLUMINANTS_SDS[illuminant]
    tristimulus_XYZ_values = []
    color_values = []
    for theta_V in u_theta_Vs:
        tristimulus_XYZ_row = []
        color_values_row = []
        for phi_V in u_phi_Vs:
            spectrum = data[u_pols == Polarization, u_theta_Is == Theta_I, u_phi_Is == Phi_I, :,u_theta_Vs == theta_V, u_phi_Vs == phi_V]
            # spectrum = spectrum[0]/np.max(spectrum)
            spectrum = np.array([u_wls,spectrum[0]]).T
            spectrum = spectrum.tolist()
            spectrum = {line[0] : line[1] for line in spectrum}
            sd = clr.SpectralDistribution(spectrum)
            sd = sd.interpolate(clr.SpectralShape(380, 830, 1))
            # print(sd)
            XYZ = clr.sd_to_XYZ(sd,cmfs,illuminant)
            sRGB = clr.XYZ_to_sRGB(XYZ/100)
            sRGB = sRGBColor(sRGB[0],sRGB[1],sRGB[2])
            RGB = list(sRGB.get_upscaled_value_tuple())
            for i in range(len(RGB)):
                if RGB[i] < 0:
                    RGB[i] = 0
                elif RGB[i] > 255:
                    RGB[i] = 255
            tristimulus_XYZ_row.append(XYZ)
            color_values_row.append(RGB)
        tristimulus_XYZ_values.append(tristimulus_XYZ_row)
        color_values.append(color_values_row)
    return tristimulus_XYZ_values, color_values

