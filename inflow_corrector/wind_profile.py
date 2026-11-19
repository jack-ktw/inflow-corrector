import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class WindProfile:
    """
    Class to store information for wind profile.

    Attributes:
        z_ref (float): reference height
        u_ref (float): reference velocity
        scale (float): geometric scale
        z_array (1-D np.array): elevations
        mean_uvw_array (3-D np.array): uvw mean velocity (reduced scale, in m/s)
        std_uvw_array (3-D np.array): uvw std velocity  (reduced scale, in m/s)
        uvw_lengthscale_array (3-D np.array): profiles of uvw mean lengthscale 
        (reduced scale, in m)
    """
    def __init__(self, z_ref=0.4, u_ref=11.8, scale=400, ls_factor=1):        
        self.z_ref = z_ref
        self.u_ref = u_ref
        self.scale = scale
        self.z_array = None
        self.mean_uvw_array = None
        self.std_uvw_array = None
        self.uvw_lengthscale_array = None
        self.ls_factor = ls_factor
    
    def set_z_array(self, z_array, normalized=False):
        if normalized:
            self.z_array = z_array * self.z_ref
        else:
            self.z_array = z_array
            
    def set_uvw_array(self, u_array, v_factor=0, w_factor=0,
                      normalized=False, clipped=True):
        """ Set u array, applying scaling factor for v and w array.

        Args:
            u_array (1-D np.array): vertical u profile
            v_factor (float, optional): scaling factor for v_array. Defaults to 0.
            w_factor (float, optional): scaling factor for w_array. Defaults to 0.
            normalized (bool, optional): True if u_array is normalized. Defaults to False.
            clipped (bool, optional): True if u_array should be clipped to prevent neg. value
        """
        self.mean_uvw_array = np.zeros((3, len(u_array)))
        
        if clipped:
            u_array = np.clip(u_array, 0.000001, None)
            
        if normalized:
            self.mean_uvw_array[0, :] = u_array * self.u_ref
        else:
            self.mean_uvw_array[0, :] = u_array
        self.mean_uvw_array[1, :] = self.mean_uvw_array[0, :] * v_factor
        self.mean_uvw_array[2, :] = self.mean_uvw_array[0, :] * w_factor

    def set_std_uvw_array(self, u_intensity_arr, iv_factor=0.78, iw_factor=0.55,
                          clipped=True):
        """ Set std_u_array, applying scaling factor for v and w intensity array.

        Args:
            u_intensity_arr (1-D np.array): vertical u intensity profile (fraction).
            iv_factor (float, optional): scaling factor for v_array. Defaults to 0.8.
            iw_factor (float, optional): scaling factor for w_array. Defaults to 0.5.
            clipped (bool, optional): True if u_intensity_arr should be clipped to prevent neg. value
        """ 

        if clipped:
            u_intensity_arr = np.clip(u_intensity_arr, 0.000001, None)
        self.std_uvw_array = np.zeros((3, len(u_intensity_arr)))
        self.std_uvw_array[0, :] = u_intensity_arr * self.mean_uvw_array[0, :]
        self.std_uvw_array[1, :] = self.std_uvw_array[0, :] * iv_factor
        self.std_uvw_array[2, :] = self.std_uvw_array[0, :] * iw_factor
    
    def set_uvw_lengthscale_array(self, u_ls_array, lv_factor=0.3, 
                                  lw_factor=0.25, normalized=False, clipped=True):
        """Set u_lengthscale_array, applying scaling factor for v and w ls array.

        Args:
            u_ls_array (1-D np.array): vertical u lengthscale profile.
            lv_factor (float, optional): scaling factor for v_array. Defaults to 0.25.
            lw_factor (float, optional): scaling factor for w_array. Defaults to 0.25.
            normalized (bool, optional): True if u_ls_array is normalized by z_ref.
            clipped (bool, optional): True if u_ls_array should be clipped to prevent neg. value
        """
        self.uvw_lengthscale_array = np.zeros((3, len(u_ls_array)))
        if clipped:
            u_ls_array = np.clip(u_ls_array, 0.000001, None)
        if normalized:
            self.uvw_lengthscale_array[0, :] = u_ls_array * self.z_ref
        else:
            self.uvw_lengthscale_array[0, :] = u_ls_array
        self.uvw_lengthscale_array[1, :] = self.uvw_lengthscale_array[0, :] * lv_factor
        self.uvw_lengthscale_array[2, :] = self.uvw_lengthscale_array[0, :] * lw_factor
        
    def set_aij_lengthscale(self, factor=1):
        aij_lu_array = 100 * (self.z_array * self.scale / 30) ** 0.5 / self.scale
        aij_lu_array[self.z_array * self.scale < 30] = 100 / self.scale
        aij_lu_array = aij_lu_array * factor
        self.set_uvw_lengthscale_array(aij_lu_array)
        
    def set_counihan_lengthscale(self, c=24.06, power=0.507, factor=1):
        # Counihan, J. (1975). Adiabatic atmospheric boundary layers: 
        # A review and analysis of data from the period 1880�?1972. Atmospheric 
        # Environment (1967), 9(10), 871�?905. 
        # https://doi.org/10.1016/0004-6981(75)90088-8

        lu_array = c * (self.z_array * self.scale) ** power / self.scale
        lu_array = lu_array * factor
        # print(lu_array, self.z_array)
        self.set_uvw_lengthscale_array(lu_array, lv_factor=0.3, lw_factor=0.25)
        
    def set_power_law(self, c=24.06, power=0.507, factor=1):
        lu_array = c * (self.get_z_array(normalized=True)) ** power
        lu_array = lu_array * factor
        self.set_uvw_lengthscale_array(lu_array, lv_factor=0.3, lw_factor=0.25)
        
    def set_constant_lengthscale(self, factor=1):
        lu_array = np.ones_like(self.z_array) * self.z_ref * 2
        lu_array = lu_array * factor
        # print(lu_array, self.z_array)
        self.set_uvw_lengthscale_array(lu_array, lv_factor=0.3, lw_factor=0.25)
        
    def set_counihan_lengthscale_v2(self, c=0.90, power=0.52, factor=1):
        # 上田�?. (1993). 変動風圧による荷重効果を�?慮した陸屋�?�構�?梁�?�風荷重に関する研究. 日本大学博士学位論文. https://cir.nii.ac.jp/crid/1570854175564417920
        # straight line fit
        lu_array = c * (self.get_z_array(normalized=True)) ** power
        lu_array = np.clip(lu_array, 0.000001, 0.628) 
        lw_array = 0.4*self.get_z_array(normalized=False)
        lw_array = np.clip(lw_array, 0.000001, 0.628*0.25)
        lv_array = 0.3 * lu_array   # just assumption
        
        self.uvw_lengthscale_array = np.zeros((3, len(lu_array)))
        self.uvw_lengthscale_array[0, :] = lu_array * factor
        self.uvw_lengthscale_array[1, :] = lv_array * factor
        self.uvw_lengthscale_array[2, :] = lw_array * factor

    def set_profile(self, norm_z_array, norm_u_array, u_intensity_arr, ls_type=0):
        """ Set the wind profile using output from powerlawfit and
            lengthscale of AIJ code.
        
        Args:
            norm_z_array: z array normalized by zref
            norm_u_array: mean u array normalized by uref
            u_intensity_arr: u intensity in fraction
        """
        
        self.set_z_array(norm_z_array, normalized=True)
        self.set_uvw_array(norm_u_array, v_factor=0, w_factor=0,
                      normalized=True)
        
        iv_factor=0.78
        iw_factor=0.55
        if ls_type == 0:
            self.set_aij_lengthscale(self.ls_factor)
        elif ls_type == 1:
            self.set_counihan_lengthscale(factor=self.ls_factor)
        elif ls_type == 2:
            self.set_constant_lengthscale(factor=self.ls_factor)
        elif ls_type == 3:
            self.set_power_law(c=2*self.z_ref, power=0.25)
        elif ls_type == 4:
            self.set_counihan_lengthscale_v2(factor=self.ls_factor)
            iv_factor=0.75
            iw_factor=0.50
            
        self.set_std_uvw_array(u_intensity_arr, iv_factor=iv_factor, iw_factor=iw_factor)
        
    def set_profile_all(self, z_array, mean_uvw_array, std_uvw_array,
                        uvw_lengthscale_array):
        self.z_array = z_array
        self.mean_uvw_array = mean_uvw_array
        self.std_uvw_array = std_uvw_array
        self.uvw_lengthscale_array = uvw_lengthscale_array
        
    def set_zero_vw(self):
        self.mean_uvw_array[1, :] = 0
        self.mean_uvw_array[2, :] = 0   

    def read_dfsr_profile(self, profile_fp):       
        my_data = np.genfromtxt(profile_fp)
        n = my_data.shape[0]
        
        self.z_array = my_data[:, 0]
        
        self.mean_uvw_array = np.zeros((3, n))
        self.mean_uvw_array[0, :] = my_data[:, 1]
        
        self.std_uvw_array = np.zeros((3, n))
        self.std_uvw_array[0, :] = my_data[:, 2] * my_data[:, 1]
        self.std_uvw_array[1, :] = my_data[:, 3] * my_data[:, 1]
        self.std_uvw_array[2, :] = my_data[:, 4] * my_data[:, 1]
        
        self.uvw_lengthscale_array = np.zeros((3, n))
        self.uvw_lengthscale_array[0, :] = my_data[:, 5]
        self.uvw_lengthscale_array[1, :] = my_data[:, 6]
        self.uvw_lengthscale_array[2, :] = my_data[:, 7]

        return my_data
    
    def scale_u_array(self, u_factor, v_factor, w_factor):
        self.mean_uvw_array[0, :] = self.mean_uvw_array[0, :] * u_factor
        self.mean_uvw_array[1, :] = self.mean_uvw_array[1, :] * v_factor
        self.mean_uvw_array[2, :] = self.mean_uvw_array[2, :] * w_factor
        self.scale_std_array(u_factor, v_factor, w_factor)
        
    def scale_std_array(self, u_factor, v_factor, w_factor):
        self.std_uvw_array[0, :] = self.std_uvw_array[0, :] * u_factor
        self.std_uvw_array[1, :] = self.std_uvw_array[1, :] * v_factor
        self.std_uvw_array[2, :] = self.std_uvw_array[2, :] * w_factor
        
    def scale_ls_array(self, u_factor, v_factor, w_factor):
        self.uvw_lengthscale_array[0, :] = self.uvw_lengthscale_array[0, :] * u_factor
        self.uvw_lengthscale_array[1, :] = self.uvw_lengthscale_array[1, :] * v_factor
        self.uvw_lengthscale_array[2, :] = self.uvw_lengthscale_array[2, :] * w_factor
        
    def scale_profile(self, u_factor=1, v_factor=1, w_factor=1,
                     iu_factor=1, iv_factor=1, iw_factor=1,
                     lu_factor=1, lv_factor=1, lw_factor=1):
        self.scale_u_array(u_factor, v_factor, w_factor)
        self.scale_std_array(iu_factor, iv_factor, iw_factor)
        self.scale_ls_array(lu_factor, lv_factor, lw_factor)
        
    def get_u_intensity_array(self):
        return self.std_uvw_array / self.mean_uvw_array[0, :]
        
    def get_z_array(self, normalized=False):
        if normalized:
            return self.z_array / self.z_ref
        else:
            return self.z_array

    def get_uvw_array(self, normalized=False):
        if normalized:
            return self.mean_uvw_array / self.u_ref
        else:
            return self.mean_uvw_array
    
    def get_ls_array(self, normalized=False):
        if normalized:
            return self.uvw_lengthscale_array / self.z_ref
        else:
            return self.uvw_lengthscale_array
        
    def get_std_array(self):
        return self.std_uvw_array

    def get_reynolds_stresses(self):
        return self.std_uvw_array**2
        
    def interpolate(self, z_array):
        '''
        Interpolate onto new grid z_array, creating new instance of wind profile
        '''
        
        new_wind_profile = WindProfile(self.z_ref, self.u_ref, self.scale)
        n = len(z_array)

        new_mean_uvw_array = np.zeros((3, n))
        for i in range(new_mean_uvw_array.shape[0]):
            f = interp1d(self.z_array, self.mean_uvw_array[i, :], fill_value="extrapolate")
            new_mean_uvw_array[i, :] = f(z_array)
            
        new_std_uvw_array = np.zeros((3, n))
        for i in range(new_std_uvw_array.shape[0]):
            f = interp1d(self.z_array, self.std_uvw_array[i, :], fill_value="extrapolate")
            new_std_uvw_array[i, :] = f(z_array)
            
        new_uvw_lengthscale_array = np.zeros((3, n))
        for i in range(new_uvw_lengthscale_array.shape[0]):
            f = interp1d(self.z_array, self.uvw_lengthscale_array[i, :], fill_value="extrapolate")
            new_uvw_lengthscale_array[i, :] = f(z_array)
            
        new_wind_profile.z_array = z_array
        new_wind_profile.mean_uvw_array = new_mean_uvw_array
        new_wind_profile.std_uvw_array = new_std_uvw_array
        new_wind_profile.uvw_lengthscale_array = new_uvw_lengthscale_array
        
        return new_wind_profile
    
    def plot_mean_u_profile(self, fp='mean_u_profile.png', exp_u_norm=None,
                            exp_z_norm=None, ylim=None):
        normalized_z_array = self.get_z_array(normalized=True)
        normalized_u_array = self.get_uvw_array(normalized=True)
        legend = ["Ux", "Uy", "Uz"]
        alphas = [1, 0.5, 0.5]
        plt.figure()
        for i in range(normalized_u_array.shape[0]):
            plt.plot(normalized_u_array[i], normalized_z_array, label=legend[i], 
                     alpha=alphas[i])  
        if not exp_u_norm is None:
            plt.plot(exp_u_norm, exp_z_norm, "kx", label="WT")
        plt.title("Mean velocity profile")
        plt.xlabel("U/Uref")
        plt.ylabel("z/Zref")
        plt.grid()
        plt.legend()
        plt.xlim(left=0)
        if ylim is None:
            plt.ylim(bottom=0)
        else:
            plt.ylim(ylim)
        plt.savefig(fp)
        plt.close()
        
    def plot_u_intensity_profile(self, fp='u_intensity_profile.png', 
                                 exp_int=None, exp_z_norm=None, ylim=None):
        normalized_z_array = self.get_z_array(normalized=True)
        u_intensity_array = self.get_u_intensity_array() * 100
        legend = ["Iu", "Iv", "Iw"]
        plt.figure()
        for i in range(u_intensity_array.shape[0]):
            plt.plot(u_intensity_array[i], normalized_z_array, label=legend[i])
        if not exp_int is None:
            plt.plot(exp_int*100, exp_z_norm, "kx", label="WT")
        plt.title("Turbulent intensity profile")
        plt.xlabel("Intensity%")
        plt.ylabel("z/Zref")
        plt.grid()
        plt.legend()
        plt.xlim(left=0)
        if ylim is None:
            plt.ylim(bottom=0)
        else:
            plt.ylim(ylim)
        plt.savefig(fp)
        plt.close()
        
    def plot_u_lengthscale_profile(self, fp='u_lengthscale_profile.png',
                                   exp_ls_norm=None, exp_z_norm=None, ylim=None):
        normalized_z_array = self.get_z_array(normalized=True)
        normalized_ls_array = self.get_ls_array(normalized=True)
        legend = ["Lu", "Lv", "Lw"]
        plt.figure()
        for i in range(normalized_ls_array.shape[0]):
            plt.plot(normalized_ls_array[i], normalized_z_array,
                     label=legend[i])
        if not exp_ls_norm is None:
            plt.plot(exp_ls_norm, exp_z_norm, "kx", label="WT")
        plt.title("Turbulent length scale profile")
        plt.xlabel("L/Zref")
        plt.ylabel("z/Zref")
        plt.grid()
        plt.legend()
        plt.xlim(left=0)
        if ylim is None:
            plt.ylim(bottom=0)
        else:
            plt.ylim(ylim)
        plt.savefig(fp)
        plt.close()

    def plot_profiles(self, suffix=""):
        self.plot_mean_u_profile(fp='mean_u_profile{}.png'.format(suffix))
        self.plot_u_intensity_profile(fp='u_intensity_profile{}.png'.format(suffix))
        self.plot_u_lengthscale_profile(fp='u_lengthscale_profile{}.png'.format(suffix))
        
    def copy(self):
            """Create a copy of the WindProfile instance."""
            new_instance = WindProfile(self.z_ref, self.u_ref, self.scale, self.ls_factor)
            new_instance.z_array = self.z_array.copy() if self.z_array is not None else None
            new_instance.mean_uvw_array = self.mean_uvw_array.copy() if self.mean_uvw_array is not None else None
            new_instance.std_uvw_array = self.std_uvw_array.copy() if self.std_uvw_array is not None else None
            new_instance.uvw_lengthscale_array = self.uvw_lengthscale_array.copy() if self.uvw_lengthscale_array is not None else None
            return new_instance

if __name__ == "__main__":
    pass
    
    
    
    
    