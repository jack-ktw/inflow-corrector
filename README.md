# Inflow Corrector

Inflow Corrector is a research prototype for calibrating synthetic atmospheric
boundary-layer inflow against target wind-tunnel profiles. The example workflow
uses simulated inlet histories to update the mean velocity, turbulence
intensity, and length scales of the inlet profile.

This README documents the inlet-only iteration path
(`InflowCorrector.iterate_inlet`). Other methods and case files are
experimental and may require additional setup; they are not presented as a
turnkey workflow or stable public API.

## Example: `scale400_powerlaw_1_4`

The example case is in
[`example/scale400_powerlaw_1_4/`](example/scale400_powerlaw_1_4/). It uses the
experimental profiles in
[`WindTunnel/rectangular_building/`](WindTunnel/rectangular_building/) and an
OpenFOAM case configured to use `DFSRTurb`.

### Requirements

- Python 3 and the Python packages `numpy`, `pandas`, `matplotlib`, `scipy`,
  and `statsmodels`.
- An OpenFOAM environment with a `DFSRTurb` executable compatible with the
  case. The included Slurm scripts use site-specific modules and paths; adapt
  them for your cluster.

There is no package installer or dependency file yet. For example, install the
Python dependencies in your environment with:

```bash
python3 -m pip install numpy pandas matplotlib scipy statsmodels
```

### Initialize the inlet profile

From the repository root, enter the case directory and make the package
importable:

```bash
cd example/scale400_powerlaw_1_4
export PYTHONPATH="$(cd ../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
python3 inflow_init.py
```

Initialization fits the example wind-tunnel mean-velocity and turbulence
intensity profiles, writes the initial OpenFOAM profile to
`constant/boundaryData/windProfile/profile`, and writes inlet sampling points.
It also clears and recreates `corrector_data/`, which contains the corrector's
saved state and plots.

### Run the solver and update the inlet

Prepare/decompose the mesh for your processor count, then run `DFSRTurb` from
the case directory. For a parallel run, for example:

```bash
decomposePar -copyZero
mpirun -np 40 DFSRTurb -parallel
```

Use the processor count configured for your case. The solver must produce inlet
history files `Ux`, `Uy`, and `Uz` under
`constant/boundaryData/windProfile/sampledData/`.

After the solver finishes, update the inlet profile:

```bash
python3 inflow_iterate.py 0.5 0.5 0.5
```

The three arguments are relaxation factors for mean velocity, turbulence
intensity, and length scale, respectively. The script reads the latest
corrector state and inlet histories, updates `profile`, and saves plots and
state under `corrector_data/`. Repeat the solver and update steps to continue
the iteration; do not rerun `inflow_init.py` between iterations, since it
resets the saved corrector state.

The included `job-inflow_init.sh` and `job-inflow_iterate.sh` are Slurm examples
for the original cluster environment. Submit them from the case directory after
setting `PYTHONPATH` as above, and adjust their module and virtual-environment
commands for your system.

## Publication

If you use this software, please cite:

Wong, J. K., Mercan, O., & Kushner, P. J. (2026). Calibration of synthetic
inflow turbulence and its effects on the wind loads on a tall building.
*Journal of Wind Engineering and Industrial Aerodynamics*, *270*, 106362.
[https://doi.org/10.1016/j.jweia.2026.106362](https://doi.org/10.1016/j.jweia.2026.106362)
