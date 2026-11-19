import os
import sys
import inflow_corrector as ic

'''
Takes relaxation factors for mean, intensity and lengthscale as input.
Update DFSR profile.
'''
if __name__ == "__main__": 
    alpha_u=float(sys.argv[1])
    alpha_i=float(sys.argv[2])
    alpha_l=float(sys.argv[3])
    ic.InflowCorrector.iterate_inlet(".", alpha_u, alpha_i, alpha_l, 0)