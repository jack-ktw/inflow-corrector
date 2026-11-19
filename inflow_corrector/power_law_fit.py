import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit

class PowerLawFit:
    '''
    Class to calculate profile as per powerlaw fitting
    phi(z_norm) = m*z_norm**alpha + c
    '''
    def calc_power_profile(m, c, alpha, z_norm_array):
        '''
        Calculate powerlaw profile given by m*z_norm**alpha + c
        '''
        if alpha < 0 and z_norm_array[0] == 0:
            phi_array = np.zeroslike(z_norm_array)
            phi_array[1:] = m * z_norm_array[1:] ** alpha + c
            phi_array[0] = phi_array[1]
        else:
            phi_array = m * z_norm_array ** alpha + c
        return phi_array

    def __init__(self):
        self.m = None
        self.c = None
        self.alpha = None
        
    def get_coeffs(self):
        return [self.m, self.c, self.alpha]

    def set_coeffs(self, m, c, alpha):
        self.m = m
        self.c = c
        self.alpha = alpha

    def calc_profile(self, z_norm_array):
        '''
        Calculate powerlaw profile given by m*z_norm**alpha + c
        '''
        return PowerLawFit.calc_power_profile(self.m, self.c, self.alpha, 
                                        z_norm_array)
    
    def calc_error(self, data_phi, data_z_norm, m=None, c=None, alpha=None):
        if m is None:
            m = self.m
        if c is None:
            c = self.c
        if alpha is None:
            alpha = self.alpha
        phi = PowerLawFit.calc_power_profile(m, c, alpha, 
                                             data_z_norm)
        diff = phi - data_phi
        return np.linalg.norm(diff, ord=2)
        
    def fit_m(self, data_phi, data_z_norm, rranges=[slice(-10, 10, 0.1)]):
        def error_func(m, *params):
            data_phi, data_z_norm = params
            return self.calc_error(data_phi, data_z_norm, m=m)

        params = (data_phi, data_z_norm)
        resbrute2 = optimize.brute(error_func, rranges, args=params, 
                                   full_output=True, finish=optimize.fmin)
        self.m = resbrute2[0]
        print("Optimal m: ", self.m)
        print("Error: ", resbrute2[1])
        return self.m
    
    def fit_m_alpha(self, data_phi, data_z_norm):
        def func(data_z_norm, m, c, alpha):
            return PowerLawFit.calc_power_profile(m, c, alpha, data_z_norm)

        lbounds = (0, -0.00001, -1)
        ubounds =  (1, 0.00001, 1)
        p0 = (self.m, self.c, self.alpha)
        popt, pcov = curve_fit(func, data_z_norm, data_phi, p0=p0,
                               bounds=(lbounds, ubounds))
        self.m = popt[0]
        self.c = popt[1]
        self.alpha = popt[2]
        print("popt:", popt)
        print("pcov:", pcov)

        return popt
    
    def fit_all(self, data_phi, data_z_norm):
        def func(data_z_norm, m, c, alpha):
            return PowerLawFit.calc_power_profile(m, c, alpha, data_z_norm)

        p0 = (self.m, self.c, self.alpha)
        popt, pcov = curve_fit(func, data_z_norm, data_phi, p0=p0)
        self.m = popt[0]
        self.c = popt[1]
        self.alpha = popt[2]
        print("popt:", popt)
        print("pcov:", pcov)

        return popt


if __name__ == "__main__":
    from exp_data import ExpData

    exp_dir = r"/project/w/wind/wongja33/WindTunnel/rectangular_building"
    ux_fp = os.path.join(exp_dir, "u_1_4.csv")
    iu_fp = os.path.join(exp_dir, "Iu_1_4.csv")
    exp_data = ExpData(ux_fp, iu_fp)
    
    norm_u_arr_exp, norm_z_arr_exp = exp_data.get_u_profile(normalized=True)
    powerlawfit = PowerLawFit()
    powerlawfit.set_coeffs(1, 0, 0.25)
    norm_u_arr_before = powerlawfit.calc_profile(norm_z_arr_exp)
    powerlawfit.fit_m(norm_u_arr_exp, norm_z_arr_exp)
    norm_u_arr = powerlawfit.calc_profile(norm_z_arr_exp)

    plt.figure()
    plt.plot(norm_u_arr_before, norm_z_arr_exp, "k--", label="before")
    plt.plot(norm_u_arr, norm_z_arr_exp, "k", label="after")
    plt.plot(norm_u_arr_exp, norm_z_arr_exp, "x", label="exp")
    plt.ylabel("z / zref")
    plt.xlabel("norm_u")
    plt.legend()
    plt.grid()
    plt.savefig(fname="powerlaw_u_profile.png")
    
    # Test for Turbulent Intensity
    print("I m alpha")
    u_intensity_arr_exp, norm_z_arr_exp = exp_data.get_intensity_profile(normalized=True)
    powerlawfit = PowerLawFit()
    powerlawfit.set_coeffs(0.116, 0, -0.30)
    u_intensity_arr_before = powerlawfit.calc_profile(norm_z_arr_exp)
    powerlawfit.fit_m_alpha(u_intensity_arr_exp, norm_z_arr_exp)
    u_intensity_arr = powerlawfit.calc_profile(norm_z_arr_exp)
    
    print("I m")
    powerlawfit2 = PowerLawFit()
    powerlawfit2.set_coeffs(0.5, 0, -0.25)
    powerlawfit2.fit_m(u_intensity_arr_exp, norm_z_arr_exp)
    u_intensity_arr2 = powerlawfit2.calc_profile(norm_z_arr_exp)
    
    print("I all")
    powerlawfit3 = PowerLawFit()
    powerlawfit3.set_coeffs(-0.5, 0, 0.30)
    powerlawfit3.fit_all(u_intensity_arr_exp, norm_z_arr_exp)
    u_intensity_arr3 = powerlawfit3.calc_profile(norm_z_arr_exp)
    
    plt.figure()
    plt.plot(u_intensity_arr_before, norm_z_arr_exp, "k--", label="before")
    plt.plot(u_intensity_arr2, norm_z_arr_exp, "C1", label="after fit_m")
    plt.plot(u_intensity_arr, norm_z_arr_exp, "k", label="after fit_m_alpha")
    plt.plot(u_intensity_arr3, norm_z_arr_exp, "C2", label="after fit_all")
    plt.plot(u_intensity_arr_exp, norm_z_arr_exp, "x", label="exp")
    plt.ylabel("z / zref")
    plt.xlabel("Iu")
    plt.legend()
    plt.grid()
    plt.savefig(fname="powerlaw_Iu_profile.png")