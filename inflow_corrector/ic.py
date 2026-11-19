import os
import re
import glob
import shutil
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from .wind_profile import WindProfile
from .exp_data import ExpData
from .power_law_fit import PowerLawFit

try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz


DEFAULT_IC_DATA_DIR = "corrector_data"
DEFAULT_IC_SAVE_DIR = "pickles"
DEFAULT_IC_PLOT_DIR = "plots"
DEFAULT_IC_PLOT_SUBDIR = "profiles"
DEFAULT_PATH_TO_EMPTY = "empty_case"
EMPTY_CASE_PROBE_RELATIVE_TOL = 0.2

class InflowCorrector:
    def __init__(self, proj_dir, z_ref, u_ref, scale, exp_ux_fp, exp_iu_fp,
                 h_ref=None, t_start=0, t_end=np.inf, ls_factor=1, ls_type=0):
        self.proj_dir = proj_dir # OpenFOAM DFSR case directory
        self.ic_data_dir = os.path.join(proj_dir, DEFAULT_IC_DATA_DIR)
        self.ic_save_dir = os.path.join(self.ic_data_dir, DEFAULT_IC_SAVE_DIR)
        self.ic_plot_dir = os.path.join(self.ic_data_dir, DEFAULT_IC_PLOT_DIR)
        self.ic_sub_dir = os.path.join(self.ic_plot_dir, DEFAULT_IC_PLOT_SUBDIR)
        self.init_data_directories()
        self.ic_empty_dir = os.path.join(self.ic_data_dir, DEFAULT_PATH_TO_EMPTY)
        self.version = "0.1"
        
        self.u_ref = u_ref
        self.z_ref = z_ref
        self.scale = scale
        
        self.ls_factor = ls_factor
        
        if h_ref == None:
            h_ref = z_ref
        self.h_ref = h_ref

        self.z_array = None
        self.ihistory_df = None
        self.ihistory_summary_df = None
        
        self.bhistory_df = None
        self.bhistory_summary_df = None

        self.specified_wind_profile = None
        self.target_wind_profile = None
        self.ihistory_profile = None
        self.bhistory_profile = None
        self.draft_wind_profile = None
        self.exp_data = ExpData(exp_ux_fp, exp_iu_fp)
        
        self.exp_ux_fp = exp_ux_fp
        self.exp_iu_fp = exp_iu_fp
        self.t_start = t_start
        self.t_end = t_end

        if not os.path.exists(self.ic_data_dir):
            os.mkdir(self.ic_data_dir)
        
        self.iteration = -1
        
    def init_data_directories(self):
        for directory in [self.ic_data_dir, self.ic_save_dir, 
                          self.ic_plot_dir, self.ic_sub_dir]:
            os.makedirs(directory, exist_ok=True)

    def set_unique_z_from_csv(self, fn="cell_xyz.csv"):
        cell_xyz_fp = os.path.join(self.proj_dir, fn)
        xz_array =  np.genfromtxt(cell_xyz_fp, delimiter=",",
                                 skip_header=1, usecols=(2, 4))
        x_array = xz_array[:, 0]
        z_array = xz_array[:, 1]
        z_array = self._round(z_array)
        z_array = np.sort(np.unique(z_array))
        z_array = InflowCorrector._pad_boundaries(z_array, lower=0, upper=self.h_ref*3.6)
        self.z_array = z_array
        
        self._min_x = np.min(x_array)

        return z_array
    
    def _round(self, z_array):
        precision = int(np.log10(self.z_ref)) + 3
        return np.round(z_array, decimals=precision)

    def _pad_boundaries(z_array, lower=0, upper=None):
        return np.concatenate([[lower], z_array, [upper]])
    
    def write_wind_profile(self, wind_profile):
        directory = os.path.join(self.proj_dir, r"constant/boundaryData/windProfile")
        os.makedirs(directory, exist_ok=True)
        profile_fn = os.path.join(self.proj_dir, r"constant/boundaryData/windProfile/profile")
        
        wp = wind_profile
        data_L = []
        data_L.append(wp.z_array)
        data_L.append(wp.mean_uvw_array[0, :])
        data_L.append(wp.std_uvw_array[0, :] / wp.mean_uvw_array[0, :])
        data_L.append(wp.std_uvw_array[1, :] / wp.mean_uvw_array[0, :])
        data_L.append(wp.std_uvw_array[2, :] / wp.mean_uvw_array[0, :])
        data_L.append(wp.uvw_lengthscale_array[0, :])
        data_L.append(wp.uvw_lengthscale_array[1, :])
        data_L.append(wp.uvw_lengthscale_array[2, :])
        np.savetxt(profile_fn, np.array(data_L).T, delimiter="\t")
    
    def write_draft_windProfile(self):
        self.write_wind_profile(self.draft_wind_profile)
        self.iteration += 1
        
    def write_samplingPoints(self):
        sub_dir = r"constant/boundaryData/windProfile/sampledData"
        directory = os.path.join(self.proj_dir, sub_dir)
        os.makedirs(directory, exist_ok=True)
        
        sub_dir = r"constant/boundaryData/windProfile/sampledData/samplingPoints"
        f_samplingPoints = os.path.join(self.proj_dir, sub_dir)

        x_min = self._min_x
        y_center = 0.000001
        with open(f_samplingPoints, "w") as f:
            f.write(str(len(self.z_array)) + "\n")
            f.write("(" + "\n")
            for z in self.z_array:
                f.write("({} {} {})\n".format(x_min, y_center, z))
            f.write(");" + "\n")
            
    def set_target_wind_profile(self, wind_profile):
        self.target_wind_profile = wind_profile
        
    def set_draft_wind_profile(self, wind_profile):
        self.draft_wind_profile = wind_profile

    def create_wind_profile_from_exp(self, ux_fp=None, iu_fp=None, ls_type=0):
        if ux_fp is None:
            ux_fp = self.exp_ux_fp
        if iu_fp is None:
            iu_fp = self.exp_iu_fp
        
        wind_profile = WindProfile(z_ref=self.z_ref, u_ref=self.u_ref,
                                   scale=self.scale, ls_factor=self.ls_factor)
        wind_profile.set_z_array(self.z_array)
        norm_z_array = wind_profile.get_z_array(normalized=True)
        
        exp_data = ExpData(ux_fp, iu_fp)
        self.exp_data = exp_data

        # Perform fitting on mean velocity
        norm_u_arr_exp, norm_z_arr_exp1 = exp_data.get_u_profile(normalized=True)
        plf_mean_u = PowerLawFit()
        plf_mean_u.set_coeffs(1, 0, 0.27)
        plf_mean_u.fit_m(norm_u_arr_exp, norm_z_arr_exp1)
        norm_u_arr = plf_mean_u.calc_profile(norm_z_array)
        
        # Perform fitting on Iu
        u_intensity_arr_exp, norm_z_arr_exp2 = exp_data.get_intensity_profile(normalized=True)
        plf_intensity = PowerLawFit()
        plf_intensity.set_coeffs(-0.5, 0, 0.30)
        plf_intensity.fit_all(u_intensity_arr_exp, norm_z_arr_exp2)
        u_intensity_arr = plf_intensity.calc_profile(norm_z_array)
        
        # Set wind_profile (the method fills lengthscale with AIJ expression)
        wind_profile.set_profile(norm_z_array, norm_u_arr, u_intensity_arr, ls_type)
        return wind_profile
        
    def read_specified_profile(self):
        profile_fp = os.path.join(self.proj_dir, r"constant/boundaryData/windProfile/profile")
        
        if not os.path.exists(profile_fp):
            print("Error: No wind profile specified.")
            return None
        
        wind_profile = WindProfile(z_ref=self.z_ref, u_ref=self.u_ref,
                                   scale=self.scale)
        wind_profile.read_dfsr_profile(profile_fp)

        self.specified_wind_profile = wind_profile
        return wind_profile
    
    def _read_inlet_history(self, fn):
        sub_dir = r"constant/boundaryData/windProfile/sampledData/"
        fp = os.path.join(self.proj_dir, sub_dir)
        fp = os.path.join(fp, fn)

        prog = re.compile(r"# (?:Point|Probe) [0-9]+ \(([0-9\.\-]+),? ([0-9\.\-]+),? ([0-9\.\-]+)\).*\n")
        x_li, y_li, z_li = [], [], []
        
        use_col_idx = [0]
        prev_coord = (-999, -999, -999)
        with open(fp, "r") as f:
            line = f.readline()
            line = f.readline()
            group = prog.match(line)
            i = 0
            while group:
                x = float(group[1])
                y = float(group[2])
                z = float(group[3])
                
                if (x, y, z) != prev_coord and line.find("Not Found") == -1:
                    x_li.append(x)
                    y_li.append(y)
                    z_li.append(z)
                    prev_coord = (x, y, z)
                    use_col_idx.append(i+1)
                    
                line = f.readline()
                group = prog.match(line)
                i += 1

        x_array = np.array(x_li)
        y_array = np.array(y_li)
        z_array = np.array(z_li)

        n_nodes = i
        inlet_history_df = pd.read_csv(fp, delimiter=r"\s+", skiprows=n_nodes+2,
                                       header=None, usecols=use_col_idx)
        inlet_history_df.columns = ["Time"] + list(range(len(use_col_idx)-1))
        inlet_history_df = inlet_history_df.reset_index(drop=True)

        return x_array, y_array, z_array, inlet_history_df
    
    def read_empty_case_history(self):
        data_dir = os.path.join(self.ic_empty_dir, r"postProcessing/InflowProbes")
        tol = EMPTY_CASE_PROBE_RELATIVE_TOL * self.h_ref

        timestep_li = [os.path.basename(x[0]) for x in os.walk(data_dir)]
        timestep_li.pop(0) # remove "."
        
        u_df_li = []
        for ts in timestep_li:
            probe_fp = os.path.join(data_dir, r"{}/U".format(ts))

            prog = re.compile(r"# (?:Point|Probe) [0-9]+ \((.+),? (.+),? (.+)\).*\n")

            x_li, y_li, z_li = [], [], []
            
            use_col_idx = [0]
            prev_coord = (-999, -999, -999)
            with open(probe_fp, "r") as f:
                line = f.readline()
                group = prog.match(line)
                i = 0
                while group:
                    x = float(group[1])
                    y = float(group[2])
                    z = float(group[3])

                    if (x, y, z) != prev_coord and \
                        abs(x) <= tol and abs(y) <= tol and abs(z) != 0 and \
                        line.find("Not Found") == -1:
                        x_li.append(x)
                        y_li.append(y)
                        z_li.append(z)
                        prev_coord = (x, y, z)
                        use_col_idx.append(i+1)
                        
                    line = f.readline()
                    group = prog.match(line)
                    i += 1

            x_array = np.array(x_li)
            y_array = np.array(y_li)
            z_array = np.array(z_li)

            n_nodes = i
            try:
                u_df = pd.read_csv(probe_fp, delimiter=r"\s{13}",
                                        skiprows=n_nodes+2, header=None,
                                        engine='python', usecols=use_col_idx)
                u_df.columns = ["Time"] + list(range(len(use_col_idx)-1))
            except pd.errors.EmptyDataError:
                u_df = pd.DataFrame()
            u_df = u_df.reset_index(drop=True)
            u_df_li.append(u_df)
            
        u_df = pd.concat(u_df_li)
        u_df = u_df.sort_values("Time")

        u_df = u_df.applymap(lambda x: x.strip("()") if isinstance(x, str) else x)
        time_df = u_df.iloc[:, 0]
        u_df = u_df.iloc[:, 1:].applymap(lambda x: x.split())

        ux_history_df = pd.concat([time_df, u_df.applymap(lambda x: float(x[0]))], axis=1)
        uy_history_df = pd.concat([time_df, u_df.applymap(lambda x: float(x[1]))], axis=1)
        uz_history_df = pd.concat([time_df, u_df.applymap(lambda x: float(x[2]))], axis=1)
        
        ux_history_df.columns = ["Time"] + list(z_array)
        
        empty_history_df = pd.melt(ux_history_df, id_vars=["Time"], var_name="z", 
                      value_name="ux")
        uy_history_df = pd.melt(uy_history_df, id_vars=["Time"], var_name="z", 
                      value_name="uy")
        uz_history_df = pd.melt(uz_history_df, id_vars=["Time"], var_name="z", 
                      value_name="uz")
        
        empty_history_df["uy"] = uy_history_df["uy"]
        empty_history_df["uz"] = uz_history_df["uz"]
        self.bhistory_df = empty_history_df
        return empty_history_df
    
    def read_inlet_histories(self):
        _, _, _, ux_history_df = self._read_inlet_history("Ux")
        _, _, _, uy_history_df = self._read_inlet_history("Uy")
        x_array, y_array, z_array, uz_history_df = self._read_inlet_history("Uz")
        
        ux_history_df.columns = ["Time"] + list(z_array)
        
        inlet_history_df = pd.melt(ux_history_df, id_vars=["Time"], var_name="z", 
                      value_name="ux")
        uy_history_df = pd.melt(uy_history_df, id_vars=["Time"], var_name="z", 
                      value_name="uy")
        uz_history_df = pd.melt(uz_history_df, id_vars=["Time"], var_name="z", 
                      value_name="uz")
        
        inlet_history_df["uy"] = uy_history_df["uy"]
        inlet_history_df["uz"] = uz_history_df["uz"]
        

        fp = os.path.join(self.ic_data_dir, "probe_locations.png")
        plt.figure()
        plt.plot(y_array, z_array, "x")
        plt.title("probe locations")
        plt.xlabel("y")
        plt.ylabel("z")
        plt.grid()
        plt.savefig(fp)
        
        self.ihistory_df = inlet_history_df
        return inlet_history_df
    
    def calculate_lengthscale(mean_u, time_array, u_array):
        def get_zero_crossing(array):
            indices = np.where(np.diff(np.sign(array)))[0]
            if len(indices) == 0 or indices[0] > len(array) - 1:
                return len(array) - 1
            else:
                return indices[0]

        r_uu_array = sm.tsa.acf(u_array, nlags=len(u_array)-1, fft=True)
        integral_array = cumtrapz(r_uu_array, x=time_array, initial=0)
        zero_crossing_index =  get_zero_crossing(r_uu_array)
        lengthscale = integral_array[zero_crossing_index-1] * mean_u

        return lengthscale
    
    def calculate_hist_statistics(self, t_start=None, t_end=None, location="inlet"):
        if t_start is None:
            t_start = self.t_start
        if t_end is None:
            t_end = self.t_end
        
        if location == "inlet":
            df = self.ihistory_df
        else:
            df = self.bhistory_df
               
        select_df = df[(t_start <= df["Time"]) & (df["Time"] < t_end)]
        summary_df = select_df.groupby("z").agg(["mean", "std", "min", "max"])
        
        # calculate lengthscale
        calculate_lengthscale = InflowCorrector.calculate_lengthscale   
        
        ls_summary_df = select_df.groupby("z").apply(lambda group: pd.Series({
            "l_u": calculate_lengthscale(group["ux"].mean(), group["Time"].to_numpy(),
                                         group["ux"].to_numpy()),
            "l_v": calculate_lengthscale(group["ux"].mean(), group["Time"].to_numpy(),
                                         group["uy"].to_numpy()),
            "l_w": calculate_lengthscale(group["ux"].mean(), group["Time"].to_numpy(),
                                         group["uz"].to_numpy())
        }))
        
        summary_df.columns = summary_df.columns.map('_'.join)
        summary_df = summary_df.join(ls_summary_df, on='z')
        
        summary_df = summary_df.reset_index()
        
        if location == "inlet":
            self.ihistory_summary_df = summary_df
        else:
            self.bhistory_summary_df = summary_df
        
        
        return summary_df
    
    def create_wind_profile_from_stats(self, summary_df=None, location="inlet"):
        if summary_df is None:
            if location == "inlet":
                summary_df = self.ihistory_summary_df
            else:
                summary_df = self.bhistory_summary_df
            
        wind_profile = WindProfile(z_ref=self.z_ref, u_ref=self.u_ref,
                                   scale=self.scale)
        
        z_array = summary_df["z"].to_numpy()
        mean_uvw_array = summary_df[["ux_mean", "uy_mean", "uz_mean"]].to_numpy().T
        std_uvw_array = summary_df[["ux_std", "uy_std", "uz_std"]].to_numpy().T
        uvw_lengthscale_array = summary_df[["l_u", "l_v", "l_w"]].to_numpy().T
        wind_profile.set_profile_all(z_array, mean_uvw_array, std_uvw_array,
                        uvw_lengthscale_array)
        
        if location == "inlet":
            self.ihistory_profile = wind_profile
        else:
            self.bhistory_profile = wind_profile
        return wind_profile

    def _smooth_array_1d(array, smooth_window):
        if smooth_window == 1:
            return array
        array_padded = np.pad(array, (smooth_window//2, smooth_window//2), mode='edge')
        kernel = np.ones(smooth_window)
        array_smoothed = np.convolve(array_padded, kernel, mode='valid')
        array_smoothed /= smooth_window
        return array_smoothed
    
    def _smooth_array(array, smooth_window):
        if smooth_window == 1:
            return array

        # Create a new array to store the smoothed values
        smoothed_array = np.zeros_like(array)

        for i in range(array.shape[0]):
            row = array[i, :]
            row_smoothed = InflowCorrector._smooth_array_1d(row, smooth_window)
            smoothed_array[i, :] = row_smoothed

        return smoothed_array
        
    def compute_updated_profile(self, smooth_window=3, alpha_u=0.5, alpha_i=0.5,
                                alpha_l=0.5, use_secant=False, use_stress=False,
                                location="inlet", use_factor=False):
        y_center = 0.000001
        
        if location == "inlet":
            history_profile = self.ihistory_profile
        else:
            history_profile = self.bhistory_profile
        
        # Interpolate all values to unique z values from inlet history
        new_z_array = history_profile.get_z_array()
        specified_wp_interp = self.specified_wind_profile.interpolate(new_z_array)
        target_wp_interp = self.target_wind_profile.interpolate(new_z_array)
        
        # Get Ux, Uy, Uz
        specified_uvw_array = specified_wp_interp.get_uvw_array()
        target_uvw_array = target_wp_interp.get_uvw_array()
        inlet_hist_uvw_array = history_profile.get_uvw_array()
        
        # Get Iu, Iv, Iw
        specified_iuvw_array = specified_wp_interp.get_u_intensity_array()
        target_iuvw_int_array = target_wp_interp.get_u_intensity_array()
        inlet_hist_iuvw_array = history_profile.get_u_intensity_array()
        
        # Get Lu, Lv, Lw
        specified_luvw_array = specified_wp_interp.get_ls_array()
        target_luvw_array = target_wp_interp.get_ls_array()
        inlet_hist_luvw_array = history_profile.get_ls_array()
        
        if use_secant:
            ic2 = InflowCorrector.open_2nd_latest(self.proj_dir)
            if ic2 is None or self.iteration < 2:
                use_secant = False
                print("ic2 is None: Using standard method")

        if use_secant:
            specified_wp_interp = ic2.specified_wind_profile.interpolate(new_z_array)
            target_wp_interp = ic2.target_wind_profile.interpolate(new_z_array)
            history_interp = ic2.history_profile.interpolate(new_z_array)
            
            # Get Ux, Uy, Uz
            specified_uvw_array2 = specified_wp_interp.get_uvw_array()
            target_uvw_array2 = target_wp_interp.get_uvw_array()
            inlet_hist_uvw_array2 = history_interp.get_uvw_array()
            
            # Get Iu, Iv, Iw
            specified_iuvw_array2 = specified_wp_interp.get_u_intensity_array()
            target_iuvw_int_array2 = target_wp_interp.get_u_intensity_array()
            inlet_hist_iuvw_array2 = history_interp.get_u_intensity_array()
            
            # Get Lu, Lv, Lw
            specified_luvw_array2 = specified_wp_interp.get_ls_array()
            target_luvw_array2 = target_wp_interp.get_ls_array()
            inlet_hist_luvw_array2 = history_interp.get_ls_array()
            
            new_uvw_array = alpha_u
            new_uvw_array *= (specified_uvw_array - specified_uvw_array2)
            new_uvw_array /= (inlet_hist_uvw_array - inlet_hist_uvw_array2)
            new_uvw_array = np.clip(new_uvw_array, 0.5, 5)
            new_uvw_array *= (target_uvw_array - inlet_hist_uvw_array)
            new_uvw_array += specified_uvw_array
            
            new_iuvw_array = alpha_i
            new_iuvw_array *= (specified_iuvw_array - specified_iuvw_array2)
            new_iuvw_array /= (inlet_hist_iuvw_array - inlet_hist_iuvw_array2)
            new_iuvw_array = np.clip(new_iuvw_array, 0.5, 5)
            new_iuvw_array *= (target_iuvw_int_array - inlet_hist_iuvw_array)
            new_iuvw_array += specified_iuvw_array
            
            new_luvw_array = alpha_l
            new_luvw_array *= (specified_luvw_array - specified_luvw_array2)
            new_luvw_array /= (inlet_hist_luvw_array - inlet_hist_luvw_array2)
            new_luvw_array = np.clip(new_luvw_array, 0.5, 5)
            new_luvw_array *= (target_luvw_array - inlet_hist_luvw_array)
            new_luvw_array += specified_luvw_array
        else:
            if use_factor:
                new_uvw_array = specified_uvw_array + alpha_u * (target_uvw_array - inlet_hist_uvw_array) * (specified_uvw_array / inlet_hist_uvw_array)
                new_iuvw_array = specified_iuvw_array + alpha_i * (target_iuvw_int_array - inlet_hist_iuvw_array) * (specified_iuvw_array / inlet_hist_iuvw_array)
                new_luvw_array = specified_luvw_array + alpha_l * (target_luvw_array - inlet_hist_luvw_array) * (specified_luvw_array / inlet_hist_luvw_array)
                
                if use_stress:
                    factor = (specified_iuvw_array*specified_uvw_array[0, :]) / (inlet_hist_iuvw_array*inlet_hist_uvw_array[0, :])
                    new_iuvw_array = specified_iuvw_array*specified_uvw_array[0, :] + \
                        alpha_i * (target_iuvw_int_array*target_uvw_array[0, :] - inlet_hist_iuvw_array*inlet_hist_uvw_array[0, :]) * factor
                    new_iuvw_array = new_iuvw_array / new_uvw_array[0, :]
            else:
                new_uvw_array = specified_uvw_array + alpha_u * (target_uvw_array - inlet_hist_uvw_array)
                new_iuvw_array = specified_iuvw_array + alpha_i * (target_iuvw_int_array - inlet_hist_iuvw_array)
                new_luvw_array = specified_luvw_array + alpha_l * (target_luvw_array - inlet_hist_luvw_array)
            
                if use_stress:
                    new_iuvw_array = specified_iuvw_array*new_uvw_array[0, :] + \
                        alpha_i * (target_iuvw_int_array*target_uvw_array[0, :] - inlet_hist_iuvw_array*inlet_hist_uvw_array[0, :])
                    new_iuvw_array = new_iuvw_array / new_uvw_array[0, :]
        
        new_uvw_array = np.clip(new_uvw_array, 0, np.inf)
        new_uvw_array[0, :] = np.clip(new_uvw_array[0, :], 0.01*self.u_ref, 2*self.u_ref)
        new_uvw_array[0, :] = np.clip(new_uvw_array[0, :], 0.8*target_uvw_array[0, :], 1.2*target_uvw_array[0, :])
        new_uvw_array = InflowCorrector._smooth_array(new_uvw_array, smooth_window)
        
        new_iuvw_array = np.clip(new_iuvw_array, 0, 1)
        new_iuvw_array = InflowCorrector._smooth_array(new_iuvw_array, smooth_window)
        
        new_luvw_array[0, :] = np.clip(new_luvw_array[0, :], 0.1*self.h_ref, 10*self.h_ref)
        new_luvw_array[1, :] = np.clip(new_luvw_array[1, :], 0.1*self.h_ref*0.3, 4*self.h_ref)
        new_luvw_array[2, :] = np.clip(new_luvw_array[2, :], 0.1*self.h_ref*0.25, 4*self.h_ref)
        new_luvw_array = InflowCorrector._smooth_array(new_luvw_array, smooth_window)
        
        new_wind_profile = WindProfile(z_ref=self.z_ref, u_ref=self.u_ref, 
                                       scale=self.scale)
        
        std_uvw_array = new_iuvw_array * new_uvw_array[0, :]

        new_wind_profile.set_profile_all(new_z_array, new_uvw_array, 
                                         std_uvw_array, new_luvw_array)
        return new_wind_profile        

    def save(self):
        dir_path = self.ic_save_dir
        fn = "inflow_corrector_{}.pkl".format(self.iteration)
        fp = os.path.join(dir_path, fn)
        with open(fp, "wb") as f:
            pickle.dump(self, f)
        print("Saved:", fp)
        
    def open_pickle(directory=".", pickle_num=0):
        dir_path = os.path.join(directory, DEFAULT_IC_DATA_DIR)
        dir_path = os.path.join(dir_path, DEFAULT_IC_SAVE_DIR)
        fp = os.path.join(dir_path, f'inflow_corrector_{pickle_num}.pkl')
        with open(fp, 'rb') as f:
            pickle_f = pickle.load(f)
        return pickle_f
        
    def open_latest(directory="."):
        dir_path = os.path.join(directory, DEFAULT_IC_DATA_DIR)
        dir_path = os.path.join(dir_path, DEFAULT_IC_SAVE_DIR)
        pickle_files = glob.glob(os.path.join(dir_path, 'inflow_corrector_*.pkl'))
        numeric_part = [int(re.search(r'\d+', os.path.basename(f)).group()) for f in pickle_files]
        index = numeric_part.index(max(numeric_part))
        latest_pickle_file = pickle_files[index]
        print(f"Opening pickle file: {latest_pickle_file}")
        with open(latest_pickle_file, 'rb') as f:
            latest_pickle = pickle.load(f)
        return latest_pickle
    
    def open_2nd_latest(directory="."):
        dir_path = os.path.join(directory, DEFAULT_IC_DATA_DIR)
        dir_path = os.path.join(dir_path, DEFAULT_IC_SAVE_DIR)
        pickle_files = glob.glob(os.path.join(dir_path, 'inflow_corrector_*.pkl'))
        
        if len(pickle_files) < 2:
            return None
        numeric_part = [int(re.search(r'\d+', os.path.basename(f)).group()) for f in pickle_files]
        sorted_numeric_part = sorted(numeric_part, reverse=True)
        second_highest_value = sorted_numeric_part[1]
        index = numeric_part.index(second_highest_value)
        sec_latest_pickle_file = pickle_files[index]
        with open(sec_latest_pickle_file, 'rb') as f:
            sec_latest_pickle = pickle.load(f)
            
        print(sec_latest_pickle_file)
        return sec_latest_pickle
    
    def clear_case(self):
        dir_path = self.ic_data_dir
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        self.init_data_directories()
        print("Cleared case...")

    def plot_profiles(self, directory=DEFAULT_IC_PLOT_DIR):
        sub_dir = self.ic_sub_dir
        
        # Create subdirectory if it does not exist
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
            
        wind_profiles = [self.specified_wind_profile, self.draft_wind_profile, 
                         self.ihistory_profile, self.target_wind_profile]
        prefixes = ["specified_", "draft_", "inlet_", "target_"]
        
        # Check if bhistory_df attribute exists, set to None if not
        if not hasattr(self, 'bhistory_df'):
            self.bhistory_df = None
            
        if self.bhistory_df is not None:
            wind_profiles.append(self.bhistory_profile)
            prefixes.append("probed_")
        
        case_suffix = ["", "_zoom"]
        ylims = [None, [0, 2]]

        for i in range(len(case_suffix)):
            fn = "mean_profiles{}_{}.png".format(case_suffix[i], self.iteration)
            exp_x, exp_y = self.exp_data.get_u_profile(normalized=True)
            for j in range(len(wind_profiles)):
                if wind_profiles[j]:
                    prefix = prefixes[j]
                    fp = os.path.join(sub_dir, prefix+fn)
                    wind_profiles[j].plot_mean_u_profile(fp, exp_x, exp_y, ylim=ylims[i])

            fn = "intensity_profiles{}_{}.png".format(case_suffix[i], self.iteration)
            exp_x, exp_y = self.exp_data.get_intensity_profile(normalized=True)
            for j in range(len(wind_profiles)):
                if wind_profiles[j]:
                    prefix = prefixes[j]
                    fp = os.path.join(sub_dir, prefix+fn)
                    wind_profiles[j].plot_u_intensity_profile(fp, exp_x, exp_y, ylim=ylims[i])
                    
            fn = "lengthscale_profiles{}_{}.png".format(case_suffix[i], self.iteration)
            for j in range(len(wind_profiles)):
                if wind_profiles[j]:
                    prefix = prefixes[j]
                    fp = os.path.join(sub_dir, prefix+fn)
                    wind_profiles[j].plot_u_lengthscale_profile(fp, ylim=ylims[i])
        print("Plotted: Profiles @", sub_dir)
        
    def plot_update(self):
        dir_path = self.ic_plot_dir

        components = ["u", "v", "w"]
        profiles = [self.specified_wind_profile, self.draft_wind_profile, 
                    self.ihistory_profile, self.target_wind_profile]
        
        labels = ["specified", "draft", "inlet_history", "target"]
        colors = ["C1", "C1", "C2", "k"]
        styles = ["--", "-", "-", "--"]
        alphas = [1, 0.9, 0.9, 1]
        
        if not hasattr(self, 'bhistory_df'):
            self.bhistory_df = None
            
        if self.bhistory_df is not None:
            profiles.append(self.bhistory_profile)
            labels.append("probed")
            colors.append("C3")
            styles.append("-")
            alphas.append(0.9)

        exp_x, exp_y = self.exp_data.get_u_profile(normalized=True)
        for i in range(len(components)):
            fn = "mean_{}_update_{}.png".format(components[i], self.iteration)
            fp = os.path.join(dir_path, fn)

            plt.figure()
            for j in range(len(profiles)):
                norm_uvw_array = profiles[j].get_uvw_array(normalized=True)
                norm_z_array = profiles[j].get_z_array(normalized=True)
                plt.plot(norm_uvw_array[i, :], norm_z_array, styles[j], 
                        color=colors[j], label=labels[j], alpha=alphas[j])
            if i == 0:
                plt.plot(exp_x, exp_y, "kx", label="WT")
            plt.plot()
            plt.grid()
            plt.title("{} Update (Iteration {})".format(components[i].upper(),
                                                        self.iteration))
            plt.xlabel("{}/Uref".format(components[i]))
            plt.ylabel("z/Zref")
            plt.legend()
            plt.savefig(fp)
            
            fn = "mean_{}_update_zoom_{}.png".format(components[i], self.iteration)
            fp = os.path.join(dir_path, fn)
            plt.title("{} Update Zoom-in (Iteration {})".format(components[i].upper(),
                                                        self.iteration))
            plt.ylim([0, 2])
            plt.xlim([0, 1.5])
            plt.savefig(fp)
            plt.close()
            
        exp_x, exp_y = self.exp_data.get_intensity_profile(normalized=True)
        for i in range(len(components)):
            fn = "intensity_{}_update_{}.png".format(components[i],
                                                     self.iteration)
            fp = os.path.join(dir_path, fn)

            plt.figure()
            for j in range(len(profiles)):
                intensity_perc_array = profiles[j].get_u_intensity_array()*100
                norm_z_array = profiles[j].get_z_array(normalized=True)
                plt.plot(intensity_perc_array[i, :], norm_z_array, styles[j], 
                        color=colors[j], label=labels[j], alpha=alphas[j])
            if i == 0:
                plt.plot(exp_x*100, exp_y, "kx", label="WT")
            plt.grid()
            plt.title("I{} Update (Iteration {})".format(components[i],
                                                         self.iteration))
            plt.xlabel("Intensity%")
            plt.ylabel("z/Zref")
            plt.legend()
            plt.xlim([0, 30])
            plt.savefig(fp)
            
            fn = "intensity_{}_update_zoom_{}.png".format(components[i],
                                                             self.iteration)
            fp = os.path.join(dir_path, fn)
            plt.title("I{} Update Zoom-in (Iteration {})".format(components[i],
                                                        self.iteration))
            plt.xlim([0, 30])
            plt.ylim([0, 2])
            plt.savefig(fp)
            plt.close()
            
        for i in range(len(components)):
            fn = "R{}_update_{}.png".format(components[i],
                                                     self.iteration)
            fp = os.path.join(dir_path, fn)

            plt.figure()
            for j in range(len(profiles)):
                std_array = profiles[j].get_std_array()
                norm_z_array = profiles[j].get_z_array(normalized=True)
                plt.plot(std_array[i, :], norm_z_array, styles[j], 
                        color=colors[j], label=labels[j], alpha=alphas[j])
            plt.grid()
            plt.title("R{} Update (Iteration {})".format(components[i],
                                                         self.iteration))
            plt.xlabel("SQRT(Reynolds stress)")
            plt.ylabel("z/Zref")
            plt.legend()
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.savefig(fp)
            
            fn = "R{}_update_zoom_{}.png".format(components[i],
                                                             self.iteration)
            fp = os.path.join(dir_path, fn)
            plt.title("R{} Update Zoom-in (Iteration {})".format(components[i],
                                                        self.iteration))
            plt.xlim(left=0)
            plt.ylim([0, 2])
            plt.savefig(fp)
            plt.close()
            
        for i in range(len(components)):
            fn = "lengthscale_{}_update_{}.png".format(components[i],
                                                       self.iteration)
            fp = os.path.join(dir_path, fn)

            plt.figure()
            for j in range(len(profiles)):
                ls_array = profiles[j].get_ls_array(normalized=True)
                norm_z_array = profiles[j].get_z_array(normalized=True)
                plt.plot(ls_array[i, :], norm_z_array, styles[j], 
                        color=colors[j], label=labels[j], alpha=alphas[j])
            plt.grid()
            plt.title("L{} Update (Iteration {})".format(components[i],
                                                         self.iteration))
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.xlabel("L/Zref")
            plt.ylabel("z/Zref")
            plt.legend()
            plt.savefig(fp)

            fn = "lengthscale_{}_update_zoom_{}.png".format(components[i],
                                                             self.iteration)
            fp = os.path.join(dir_path, fn)
            plt.title("L{} Update Zoom-in (Iteration {})".format(components[i],
                                                        self.iteration))
            plt.xlim(left=0)
            plt.ylim([0, 2])
            plt.savefig(fp)
            plt.close()

        print("Plotted: Profiles Update @", dir_path)
        
    def plot_update2(self):
        dir_path = self.ic_plot_dir

        components = ["u", "v", "w"]
        profiles = [self.specified_wind_profile, self.draft_wind_profile, 
                    self.ihistory_profile, self.target_wind_profile]
        labels = [r"$\phi^{k}$", r"$\phi^{k+1}", r"$\widehat{\phi}^{k}$", r"$\phi_{target}$"]
        colors = ["C0", "C0", "C1", "k"]
        styles = ["--", "-", "-", "--"]
        alphas = [1, 0.9, 0.9, 1]
        
        if not hasattr(self, 'bhistory_df'):
            self.bhistory_df = None
        
        if self.bhistory_df is not None:
            profiles.append(self.bhistory_profile)
            labels.append("probed")
            colors.append("C3")
            styles.append("-")
            alphas.append(0.9)

        exp_x, exp_y = self.exp_data.get_u_profile(normalized=True)
        for i in range(len(components)):
            fn = "mean_{}_update_{}_paper.png".format(components[i], self.iteration)
            fp = os.path.join(dir_path, fn)

            plt.figure()
            for j in range(len(profiles)):
                norm_uvw_array = profiles[j].get_uvw_array(normalized=True)
                norm_z_array = profiles[j].get_z_array(normalized=True)
                plt.plot(norm_uvw_array[i, :], norm_z_array, styles[j], 
                        color=colors[j], label=labels[j], alpha=alphas[j])
            if i == 0:
                plt.plot(exp_x, exp_y, "kx", label="WT")
            plt.plot()
            plt.grid()
            plt.title("{} Update (Iteration {})".format(components[i].upper(),
                                                        self.iteration))
            plt.xlabel("{}/Uref".format(components[i]))
            plt.ylabel("$z/Z_{ref}$")
            plt.legend()
            plt.savefig(fp)
            
            fn = "mean_{}_update_zoom_{}_paper.png".format(components[i], self.iteration)
            fp = os.path.join(dir_path, fn)
            plt.title("{} Update Zoom-in (Iteration {})".format(components[i].upper(),
                                                        self.iteration))
            plt.ylim([0, 2])
            plt.savefig(fp)
            plt.close()
            
        exp_x, exp_y = self.exp_data.get_intensity_profile(normalized=True)
        for i in range(len(components)):
            fn = "intensity_{}_update_{}_paper.png".format(components[i],
                                                     self.iteration)
            fp = os.path.join(dir_path, fn)

            plt.figure()
            for j in range(len(profiles)):
                intensity_perc_array = profiles[j].get_u_intensity_array()*100
                norm_z_array = profiles[j].get_z_array(normalized=True)
                plt.plot(intensity_perc_array[i, :], norm_z_array, styles[j], 
                        color=colors[j], label=labels[j], alpha=alphas[j])
            if i == 0:
                plt.plot(exp_x*100, exp_y, "kx", label="WT")
            plt.grid()
            plt.title("I{} Update (Iteration {})".format(components[i],
                                                         self.iteration))
            plt.xlabel("Intensity%")
            plt.ylabel("$z/Z_{ref}$")
            plt.legend()
            plt.xlim([0, 30])
            plt.savefig(fp)
            
            fn = "intensity_{}_update_zoom_{}_paper.png".format(components[i],
                                                             self.iteration)
            fp = os.path.join(dir_path, fn)
            plt.title("I{} Update Zoom-in (Iteration {})".format(components[i],
                                                        self.iteration))
            plt.ylim([0, 2])
            plt.savefig(fp)
            plt.close()
            
        for i in range(len(components)):
            fn = "lengthscale_{}_update_{}_paper.png".format(components[i],
                                                       self.iteration)
            fp = os.path.join(dir_path, fn)

            plt.figure()
            for j in range(len(profiles)):
                ls_array = profiles[j].get_ls_array(normalized=True)
                norm_z_array = profiles[j].get_z_array(normalized=True)
                plt.plot(ls_array[i, :], norm_z_array, styles[j], 
                        color=colors[j], label=labels[j], alpha=alphas[j])
            plt.grid()
            plt.title("L{} Update (Iteration {})".format(components[i],
                                                         self.iteration))
            plt.xlabel("L/Zref")
            plt.ylabel("$z/Z_{ref}$")
            plt.legend()
            plt.savefig(fp)

            fn = "lengthscale_{}_update_zoom_{}_paper.png".format(components[i],
                                                             self.iteration)
            fp = os.path.join(dir_path, fn)
            plt.title("L{} Update Zoom-in (Iteration {})".format(components[i],
                                                        self.iteration))
            plt.ylim([0, 2])
            plt.savefig(fp)
            plt.close()

        print("Plotted: Profiles Update @", dir_path)
            
    def restart(self, ls_type=0):
        '''
        Reset input profile using exp data
        '''
        self.clear_case()
        self.set_unique_z_from_csv(fn="cell_xyz.csv")
        if self.exp_iu_fp == None:
            wind_profile = self.read_specified_profile()
        else:
            wind_profile = self.create_wind_profile_from_exp(ls_type=ls_type)
        self.set_target_wind_profile(wind_profile.copy())
        self.set_draft_wind_profile(wind_profile)
        self.write_draft_windProfile()
        self.plot_profiles()
        self.save()
        self.write_samplingPoints()
        
    def restart_from_current(self, ls_type=0):
        '''
        Keep the current input profile
        '''
        self.clear_case()
        self.set_unique_z_from_csv(fn="cell_xyz.csv")
        wind_profile = self.create_wind_profile_from_exp(ls_type=ls_type)
        self.set_target_wind_profile(wind_profile)
        wind_profile = self.read_specified_profile()
        self.set_draft_wind_profile(wind_profile)
        self.write_draft_windProfile()
        self.plot_profiles()
        self.save()
        self.write_samplingPoints()
        
    def write_draft_from_pkl(proj_dir):
        ic = InflowCorrector.open_latest(proj_dir)
        ic.write_draft_windProfile()
        ic.plot_profiles()
        print("overwriting profiles")
        
    def smooth_draft_from_pkl(proj_dir, smooth_window=3):
        ic = InflowCorrector.open_latest(proj_dir)
        
        draft_wind_profile = ic.draft_wind_profile
        draft_uvw_array = draft_wind_profile.get_uvw_array()
        draft_iuvw_array = draft_wind_profile.get_u_intensity_array()
        draft_luvw_array = draft_wind_profile.get_ls_array()
        
        z_array = draft_wind_profile.get_z_array()
        new_uvw_array = InflowCorrector._smooth_array(draft_uvw_array, smooth_window)
        new_iuvw_array = InflowCorrector._smooth_array(draft_iuvw_array, smooth_window)
        new_luvw_array = InflowCorrector._smooth_array(draft_luvw_array, smooth_window)

        new_wind_profile = WindProfile(z_ref=ic.z_ref, u_ref=ic.u_ref, 
                                       scale=ic.scale)
        
        std_uvw_array = new_iuvw_array * new_uvw_array[0, :]

        new_wind_profile.set_profile_all(z_array, new_uvw_array, 
                                         std_uvw_array, new_luvw_array)
        ic.set_draft_wind_profile(new_wind_profile)
        ic.iteration -= 1
        ic.plot_update()
        ic.plot_update2()
        ic.write_draft_windProfile()
        ic.plot_profiles()
        ic.save()
        print("overwriting profiles")
        
    def iterate_inlet(proj_dir, alpha_u=0.5, alpha_i=0.5, alpha_l=0.5,
                read_specified_profile=True, use_secant=False, use_stress=False, use_factor=False):
        ic = InflowCorrector.open_latest(proj_dir)
        ic.init_data_directories()
        if read_specified_profile:
            print("Reading Wind Profile as Specified Profile")
            ic.read_specified_profile()
        else:
            print("Using Draft Wind Profile as Specified Profile")
            ic.specified_wind_profile = ic.draft_wind_profile.copy()
        ic.read_inlet_histories()
        ic.calculate_hist_statistics(location="inlet")
        ic.create_wind_profile_from_stats()
        wind_profile = ic.compute_updated_profile(alpha_u=alpha_u, alpha_i=alpha_i,
                                                  alpha_l=alpha_l, use_secant=use_secant,
                                                  use_stress=use_stress, use_factor=use_factor)
        wind_profile.set_zero_vw()
        ic.set_draft_wind_profile(wind_profile)
        ic.plot_update()
        ic.write_draft_windProfile()
        ic.plot_profiles()
        ic.save()
        
    def set_link_to_empty_case(proj_dir, empty_dir):
        ic = InflowCorrector.open_latest(proj_dir)
        ic.ic_data_dir = ic.ic_data_dir.replace(r"./", proj_dir + r"/")
        ic.ic_save_dir = ic.ic_save_dir.replace(r"./", proj_dir + r"/")
        ic.ic_plot_dir = ic.ic_plot_dir.replace(r"./", proj_dir + r"/")
        ic.ic_sub_dir = ic.ic_sub_dir.replace(r"./", proj_dir + r"/")
        ic.proj_dir = ic.proj_dir.replace(r"./", proj_dir + r"/")
        
        ic.ic_empty_dir = os.path.join(ic.ic_data_dir, DEFAULT_PATH_TO_EMPTY)
        ic.link_empty_case(empty_dir)
        print("Saving empty_dir link to ic:")
        ic.save()
        
    def iterate_empty(proj_dir, alpha_u=0.5, alpha_i=0.5, alpha_l=0.5,
                read_specified_profile=True, use_secant=False, use_stress=False,
                use_factor=False):
        ic = InflowCorrector.open_latest(proj_dir)
        ic.init_data_directories()
        if read_specified_profile:
            print("Reading Wind Profile as Specified Profile")
            ic.read_specified_profile()
        else:
            print("Using Draft Wind Profile as Specified Profile")
            ic.specified_wind_profile = ic.draft_wind_profile.copy()
            
        # read inlet histories and calculate statistics
        ic.read_inlet_histories()
        ic.calculate_hist_statistics(location="inlet")
        ic.create_wind_profile_from_stats(location="inlet")

        # read time histories at inflowProbes and calculate statistics
        ic.read_empty_case_history()
        ic.calculate_hist_statistics(location="building")
        ic.create_wind_profile_from_stats(location="building")
        wind_profile = ic.compute_updated_profile(alpha_u=alpha_u, alpha_i=alpha_i,
                                                  alpha_l=alpha_l, use_secant=use_secant,
                                                  use_stress=use_stress, location="building", use_factor=use_factor)
        wind_profile.set_zero_vw()
        ic.set_draft_wind_profile(wind_profile)
        ic.plot_update()
        ic.write_draft_windProfile()
        ic.plot_profiles()
        ic.save()
        
    def scale_inflow(proj_dir, u_factor=1, v_factor=1, w_factor=1,
                     iu_factor=1, iv_factor=1, iw_factor=1,
                     lu_factor=1, lv_factor=1, lw_factor=1):
        ic = InflowCorrector.open_latest(proj_dir)
        
        print("Scaling draft profile and target profile:")
        print(f"uvw_factors: {u_factor}, {v_factor}, {w_factor}")
        print(f"iuvw_factors: {iu_factor}, {iv_factor}, {iw_factor}")
        print(f"luvw_factors: {lu_factor}, {lv_factor}, {lw_factor}")
        profiles = [ic.draft_wind_profile, ic.target_wind_profile]
        for profile in profiles:
            profile.scale_profile(u_factor, v_factor, w_factor, 
                                iu_factor, iv_factor, iw_factor,
                                lu_factor, lv_factor, lw_factor)
            
        print("Writing updated draft profile")
        ic.write_draft_windProfile()
        ic.plot_profiles()
        ic.save()
        
    def link_empty_case(self, case_dir):
        try:
            os.unlink(self.ic_empty_dir)
        except FileNotFoundError:
            pass
        os.symlink(case_dir, self.ic_empty_dir)
        
    def write_probes(proj_dir):
        ic = InflowCorrector.open_latest(proj_dir)
        ic.set_unique_z_from_csv()
        x_li = [-2.5, 0]
        with open("probes_coords.txt", "w") as f:
            for x in x_li:
                for z in ic.z_array:
                    f.write("({:.6f} {:.6f} {:.6f})\n".format(x, 0, z))
        print("dumping out coordinates")
        
if __name__ == "__main__":
    pass
    
    
    
    
    
    
    
    
    