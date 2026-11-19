import os
import inflow_corrector as ic

'''
Initialize the DFSR profile.
'''
if __name__ == "__main__": 
    proj_dir = r"."
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    exp_dir = os.path.join(repo_dir, "WindTunnel", "rectangular_building")
    ux_fp = os.path.join(exp_dir, "u_1_4.csv")
    iu_fp = os.path.join(exp_dir, "Iu_1_4.csv")

    corrector = ic.InflowCorrector(proj_dir, 0.4, 11.8, 400, ux_fp, iu_fp,
                            h_ref=0.5, t_start=5, t_end=10, ls_factor=1)
    corrector.restart()