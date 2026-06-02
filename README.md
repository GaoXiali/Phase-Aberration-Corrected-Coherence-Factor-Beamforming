# PAC-CF Reproduction Package

This repository provides a compact MATLAB reproduction package for selected PAC-CF reconstruction and simulation results associated with the manuscript. It is intended for reviewers and readers to run the supplied demos and inspect the visual behavior of DAS, conventional CF, and PAC-CF under the provided experimental and synthetic settings.

This is not a full development toolbox. The CUDA source code is not included; the reconstruction demos rely on precompiled Windows MEX binaries.

## Tested Environment

The code has been tested on:

- GPU: NVIDIA RTX 4090
- CUDA: 12.9
- MATLAB: R2024a
- Platform: Windows, using `.mexw64` binaries

The MEX files in `src/` are sensitive to the CUDA, driver, MATLAB, and GPU runtime environment. If your CUDA version or MATLAB/MEX runtime differs from the tested setup, the MEX functions may fail to load or execute. In that case, use the tested environment or rebuild compatible MEX binaries from your own source implementation.

## Layout

- `mouse/code/`: mouse reconstruction entry scripts for the selected full-body, liver-region, and mesenteric-artery demos.
- `phantom/code/`: phantom reconstruction entry script comparing DAS, CF, and PAC-CF.
- `simulation/code/`: synthetic PAC-CF simulation scripts for principle and boundary-error studies.
- `src/`: shared MATLAB helpers, detector coordinates, and precompiled MEX runtime files.
- `mouse/data/raw/`: raw dual-side mouse dataset.
- `phantom/data/raw/`: raw phantom dataset.

## How To Run

Open MATLAB from this repository root and run one entry script, for example:

```matlab
run('mouse/code/SphericalRecon_mouse_mesenteric_artery.m')
run('phantom/code/SphericalRecon_phantom.m')
run('simulation/code/run_paccf_principle_sweep.m')
```

The scripts are intended for visual reproduction. They display figures and print metrics where applicable, but they do not save reconstructed volumes, generated figures, CSV files, or reports.

## Data And Large Files

The package includes large `.mat` inputs. When publishing to GitHub or another Git remote, store `.mat` files with Git LFS or host them in a data archive and link them from this README.

## Citation

If you use this package, please cite the associated manuscript and this repository.
