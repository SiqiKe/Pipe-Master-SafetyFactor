# Drainage Network Design Under Uncertainty

This repository provides a workflow for stormwater pipe network design under deep uncertainty proposed in the working paper "A safety factor approach to navigate cost-reliability trade-offs in designing urban drainage systems." It integrates climate change projections, land-use change, and infrastructure aging to identify and evaluate robust design strategies.

## Paper Abstract

Designing urban stormwater drainage systems poses challenges. For one, projections of critical drivers of system performance, such as precipitation increases, land use changes, and infrastructure degradation, are deeply uncertain. Moreover, decision-makers face trade-offs between multiple objectives, such as reducing costs and improving reliability. Furthermore, when a drainage network contains many individual pipes, the large number of decision levers can lead to substantial computational demands. While previous research has established effective design frameworks, they typically remain silent on the interactions between (i) the effect of deep uncertainty, (ii) trade-offs between design objectives, and (iii) the challenges of tackling large pipe networks. Here we use a case study to demonstrate that, under future changes in extreme precipitation intensity, impervious area, and pipe characteristics with associated uncertainties, network designs according to a federal guideline can fail to achieve their stated design objectives. As a benchmark, we identify approximately Pareto-optimal designs that balance cost, reliability, overflow volume, and robustness to deep uncertainty, using a multi-objective evolutionary algorithm to optimize the sizes of all pipes in the pipe network. We further develop safety-factor approaches that can provide Pareto improvements over a current federal guideline. Our safety-factor approaches (i) closely approximate the benchmark obtained via the computationally expensive optimization method, (ii) are conceptually simple and easy to interpret, and (iii) are computationally fast, enabling them to scale well to large networks. The safety factor approaches can provide practical guidelines for designing real-world drainage systems.


## Overview of this repo

The project evaluates multiple optimization/design approaches for stormwater systems across many uncertain futures (States of the World, SOWs). The workflow supports:

- Generating baseline designs following federal guidelines and optimized designs using different approaches discussed in the paper
- Testing designs' reliability and robustness across climate, land use, and pipe aging scenarios
- Comparing methods in objective space (reliability, robustness, flooding, and cost) and computational cost
- Performing sensitivity analysis using Sobol' method.

## Design Approaches Included

- UDD (Urban Drainage Design) standard method
- UDD with a 20% increased design storm (`UDD_20prec`)
- Pipe-by-Pipe optimization via NSGA-II (`PbP`)
- System-wide Safety Factor (`SystemSF`)
- Functional Safety Factor (`FunctionSF`)

## Project Structure

```text
pipe_network_master/
├── src/
│   ├── workflows/                    # Main end-to-end workflow scripts
│   │   ├── 1_UDD_design.py
│   │   ├── 1_UDD_design_20prec.py
│   │   ├── 2_SOW_n_conv.py
│   │   ├── 3_SOW_prep_udd_eval.py
│   │   ├── 4_BruteForce.py
│   │   ├── 5_PipeByPipe.py
│   │   ├── 6_SystemSF.py
│   │   ├── 7_FunctionSF.py
│   │   └── 8_Sobol_analysis.py
│   └── pnetwork/                     # Core package modules
├── data/
│   ├── processed/
│   └── raw/
├── results/                          # Design/evaluation outputs
├── Figures/                          # Figure notebooks and plots
├── Run_analysis.sh                   # Example batch workflow commands
└── README.md
```

## Quick Start

### Code and Data Setup

1. Clone this repository.
2. Download the data from the [Zenodo repository](https://doi.org/10.5281/zenodo.23424004).
3. Extract the relevant archives into the locations below. Keep the archive contents and filenames unchanged:
  - **a.** Extract `projection.zip` into `data/processed/precipitation/`. This is the precomputed precipitation-projection input.
  - **b.** For figure-only reproduction, extract `SOWs_output.zip` into `results/`. This optional archive provides precomputed results; no additional point locations need to be configured.
4. Navigate to the project directory.



### Virtual Environment Setup

**Important:** You must have Python 3.10 or higher installed on your system before proceeding. To check your Python version:

```bash
python --version
# or
python3 --version
```

Create a new Python virtual environment named '.venv' using Python >= 3.10 and activate it. (If you choose a different name, update the environment activation command in the "Run_analysis.sh" file referenced in the next step.) We recommend using the built-in venv tool in Python for simplicity, but you can also use conda or others.

```bash
python -m venv .venv
source .venv/bin/activate
```

### Dependency Installation
Once your virtual environment is activated, install dependencies from pyproject.toml:

```bash
pip install --upgrade pip
pip install .
```
Alternatively, if using requirements.txt:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```


### Quick Start

Run the example analysis pipeline for a 5-pipe network:

```bash
bash Run_analysis.sh
```

The script includes example commands to run all workflows. By default, evaluations run sequentially. If your computing resources allow, you can add the `--parallel` flag and increase `--num_processes` to speed up the analysis. Estimated runtime on a personal computer is ~1-2 hours for the 5-pipe network.

### Reproducing Figures

1. Register the current virtual enviroment as a Jupyter notebook kernel 'Python (PNM)'

```bash
python -m ipykernel install --user --name=env-PNM --display-name="Python (PNM)"
```

2. Place the provided results ("SOWs_output.zip") under `results/` and unzip it. Run the notebooks in `Figures/` using the kernel 'Python (PNM).' You can open the Jupyter Notebook by running the following command in the terminal (Do not close the terminal window while you are working in Jupyter) or using any method you prefer.

```bash
jupyter notebook
```

Main notebooks include:

- `Figures/Figures.ipynb` for figures in the paper
- `Figures/Figures_SI.ipynb` for figures in the supporting information

## Notes on Compute

- Some workflow steps are computationally intensive and designed for parallel execution.
- Runtime depends on network size, number of SOWs, and optimization settings.
- Reproducing the full analysis from the paper required weeks of HPC runtime, so this repository provides an example run on a small network. You are welcome to test and extend it.
- **Testing Status:**
  - ✓ Linux: Fully tested and working
  - ⚠ macOS: Has unresolved import issue with the pyswmm package
  - ✗ Windows: Not tested

## AI statement
The author used Copilot Pro to assist with documentation. 

## License

MIT License. See `LICENSE` for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{Ke2026,
  author = {Ke, Siqi},
  title = {Pipe Network Design Under Uncertainty},
  year = {2026},
  howpublished = {\url{https://github.com/SiqiKe/pipe_network_master}},
  note = {Version 1.0}
}
```

## Contact
**Siqi Ke**  
Email: siqi.ke.th@dartmouth.edu
Dartmouth College


## Acknowledgments

- EPA SWMM development team
- Climate projection source repository: https://github.com/juliangus12/CSO_Design_Optimization/tree/main/PipeDesign-master
- Python communities (`pymoo`, `pyswmm`, `SALib`)

