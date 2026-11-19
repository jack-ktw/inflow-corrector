import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit

class LogLawFit:
    '''
    Class to calculate profile as per loglaw fitting
    '''
    def calc_log_profile(z0_norm, z_norm_array):
        '''
        Calculate log profile given by u=u*/kappa*ln(z-d+z0 / z0)
        '''
        
        kappa = 0.41
        u_roughness = 1 * kappa / np.log((1 + z0_norm)/z0_norm)
        phi_array = u_roughness / kappa * np.log((z_norm_array - z0_norm) / z0_norm)

        return phi_array
    
    def calc_intensity_profile(self, z_norm_array, u_norm_array):
        '''
        Calculate log profile given by u=u*/kappa*ln(z-d+z0 / z0)
        '''
        z0_norm = self.z0_norm
        kappa = 0.41
        Cmu = 0.09
        u_roughness = 1 * kappa / np.log((1 + z0_norm)/z0_norm)
        k_array = u_roughness**2 / Cmu**0.5 * np.ones_like(z_norm_array)
        phi_array = (2/3 * k_array)**0.5 / u_norm_array

        return phi_array

    def __init__(self):
        self.z0_norm = None
        
    def get_coeffs(self):
        return [self.z0_norm]

    def set_coeffs(self, z0_norm):
        self.z0_norm = z0_norm

    def calc_profile(self, z_norm_array):
        '''
        Calculate loglaw profile
        '''
        return LogLawFit.calc_log_profile(self.z0_norm, z_norm_array)
    
    def calc_error(self, data_phi, data_z_norm, z0_norm=None):
        if z0_norm is None:
            z0_norm = self.z0_norm
        phi = LogLawFit.calc_log_profile(z0_norm, data_z_norm)
        diff = phi - data_phi
        return np.linalg.norm(diff, ord=2)
           
    def fit_z0(self, data_phi, data_z_norm):
        def func(data_z_norm, z0_norm):
            return LogLawFit.calc_log_profile(z0_norm, data_z_norm)

        lbounds = (0.000001)
        ubounds =  (1)
        p0 = (self.z0_norm)
        popt, pcov = curve_fit(func, data_z_norm, data_phi, p0=p0,
                               bounds=(lbounds, ubounds))
        self.z0_norm = popt[0]
        print("popt:", popt)
        print("pcov:", pcov)

        return popt


if __name__ == "__main__":
    from exp_data import ExpData

    exp_dir = r"../WindTunnel/rectangular_building"
    ux_fp = os.path.join(exp_dir, "u_1_4.csv")
    iu_fp = os.path.join(exp_dir, "Iu_1_4.csv")
    exp_data = ExpData(ux_fp, iu_fp)
    
    norm_u_arr_exp, norm_z_arr_exp = exp_data.get_u_profile(normalized=True)
    loglawfit = LogLawFit()
    loglawfit.set_coeffs(0.00413901)
    norm_u_arr_before = loglawfit.calc_profile(norm_z_arr_exp)
    loglawfit.fit_z0(norm_u_arr_exp, norm_z_arr_exp)
    print(loglawfit.z0_norm*0.5)
    norm_u_arr = loglawfit.calc_profile(norm_z_arr_exp)

    plt.figure()
    plt.plot(norm_u_arr_before, norm_z_arr_exp, "k--", label="before")
    plt.plot(norm_u_arr, norm_z_arr_exp, "k", label="after")
    plt.plot(norm_u_arr_exp, norm_z_arr_exp, "x", label="exp")
    plt.ylabel("z / zref")
    plt.xlabel("norm_u")
    plt.legend()
    plt.grid()
    plt.savefig(fname="loglaw_u_profile.png")
    
    # Test for Turbulent Intensity
    print("I m alpha")
    u_intensity_arr_exp, norm_z_arr_exp = exp_data.get_intensity_profile(normalized=True)
    loglawfit0 = LogLawFit()
    loglawfit0.set_coeffs(0.00413901)
    norm_u_arr = loglawfit0.calc_profile(norm_z_arr_exp)
    u_intensity_arr_before = loglawfit.calc_intensity_profile(norm_z_arr_exp, norm_u_arr)
    
    norm_u_arr = loglawfit.calc_profile(norm_z_arr_exp)
    u_intensity_arr = loglawfit.calc_intensity_profile(norm_z_arr_exp, norm_u_arr)
    
    plt.figure()
    plt.plot(u_intensity_arr_before, norm_z_arr_exp, "k--", label="before")
    # plt.plot(u_intensity_arr2, norm_z_arr_exp, "C1", label="after fit_m")
    plt.plot(u_intensity_arr, norm_z_arr_exp, "k", label="after fit")
    # plt.plot(u_intensity_arr3, norm_z_arr_exp, "C2", label="after fit_all")
    plt.plot(u_intensity_arr_exp, norm_z_arr_exp, "x", label="exp")
    plt.xlim([0, 0.3])
    plt.ylabel("z / zref")
    plt.xlabel("Iu")
    plt.legend()
    plt.grid()
    plt.savefig(fname="loglaw_Iu_profile.png")