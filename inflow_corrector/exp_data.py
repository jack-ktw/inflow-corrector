import os
import numpy as np
import pandas as pd

class DataPoints:
    def  __init__(self, data_array, z_array):
        self.data_array = data_array
        self.z_array = z_array
        
class SpectrumDataPoints:
    def  __init__(self, fr_array, su_array):
        self.fr_array = fr_array
        self.su_array = su_array
        
class ExpData:
    def __init__(self, exp_ux_fp, exp_iu_fp,
                 z_ref=0.4, u_ref=11.8):
        self.z_ref = z_ref
        self.u_ref = u_ref

        self.ux_data_points = DataPoints(np.array([]), np.array([]))
        self.iu_data_points = DataPoints(np.array([]), np.array([]))
        
        if not exp_ux_fp is None:
            self.ux_data_points = self.read_exp_profile_ux(exp_ux_fp)
        if not exp_iu_fp is None:
            self.iu_data_points = self.read_exp_profile_iu(exp_iu_fp)
            
        # self.Su_data_points = self.read_exp_profile_su()
        
    def get_u_profile(self, normalized=False):
        if normalized:
            return (self.ux_data_points.data_array / self.u_ref,
                    self.ux_data_points.z_array / self.z_ref)
        else:
            return (self.ux_data_points.data_array,
                    self.ux_data_points.z_array)
            
    def get_intensity_profile(self, normalized=False):
        if normalized:
            return (self.iu_data_points.data_array,
                    self.iu_data_points.z_array / self.z_ref)
        else:
            return (self.iu_data_points.data_array,
                    self.iu_data_points.z_array)

    def read_exp_profile_ux(self, exp_fp):
        header = ['U/Uref', 'Z/Zref']
        u_exp_df = pd.read_csv(exp_fp, names=header)
        ux_data_points = DataPoints(
            u_exp_df["U/Uref"].to_numpy() * self.u_ref,
            u_exp_df["Z/Zref"].to_numpy() / 100)
        return ux_data_points
        
    def read_exp_profile_iu(self, iu_exp_fp):
        header = ['Iu%', 'Z/Zref']
        iu_exp_df = pd.read_csv(iu_exp_fp, names=header)
        iu_data_points = DataPoints(
            iu_exp_df['Iu%'].to_numpy() / 100,
            iu_exp_df['Z/Zref'].to_numpy() / 100)
        return iu_data_points

    def read_exp_profile_su(self, fn="spectrum.csv"):
        su_exp_fp = os.path.join(self.exp_dir, fn)
        su_exp_df = pd.read_csv(su_exp_df)
        su_exp_df.columns = ['fr', 'Su']
        su_exp_df["fr"] = 10 ** su_exp_df["fr"]
        su_exp_df["Su"] = 10 ** su_exp_df["Su"]
        su_data_points = SpectrumDataPoints(su_exp_df["fr"].to_numpy(),
                                                 su_exp_df["Su"].to_numpy())
        return su_data_points
    
if __name__ == "__main__":
    exp_dir = r"../WindTunnel/rectangular_building"
    ux_fp = os.path.join(exp_dir, "u_1_4.csv")
    iu_fp = os.path.join(exp_dir, "Iu_1_4.csv")
    exp_data = ExpData(ux_fp, iu_fp)

    print("done")