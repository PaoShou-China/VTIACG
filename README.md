# VTIACG

This repository contains Python simulation and plotting scripts organized in numbered directories. The original numbering is preserved because the directories share similarly named files and the intended conceptual labels are not encoded in the code.

## Repository map

| Directory | Contents visible in the repository |
| --- | --- |
| `0/` | Standalone missile / virtual-target simulation and trajectory plotting in `main.py`. |
| `1/` | Simulation, batch execution, parameter search, comparison, and plotting scripts. |
| `2/` | Simulation, batch execution, parameter and seed/state search, comparison, and plotting scripts. |
| `3/` | Simulation, comparison, and plotting scripts. |

The numbered folders are separate script collections. Their files with the same name are not necessarily interchangeable. Run a script from its own directory unless the script itself specifies otherwise.

## Requirements

The inspected scripts use Python, NumPy, and Matplotlib. Some later experiment scripts may require additional packages; install any package named by a script's imports if it is missing.

A basic environment can be prepared with:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy matplotlib
```

## Running scripts

For example:

```bash
cd 0
python main.py
```

For a script in another folder, change into that folder and invoke its filename, such as `python simulation.py` or `python batch.py`. Inspect the script's `if __name__ == "__main__"` block before running: several scripts launch simulations, parameter searches, or plots directly and may take longer than a single demonstration.

Most parameters are currently defined in the scripts. Review the source before changing them. Plot windows may require a graphical desktop; configure a non-interactive Matplotlib backend when running headlessly.

## Reproducibility notes

Some simulation files set NumPy's global random seed in module scope; others may use different or no seed handling. Record the script, parameters, Python/package versions, and seed alongside any result you use. This repository does not currently provide a single shared configuration or unified runner.
