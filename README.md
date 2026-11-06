# VTIACG

**VTIACG** is a set of Python scripts for two-dimensional missile and virtual-target guidance simulations, parameter studies, comparisons, and plots. The numbered folders preserve the existing experiment organization.

[Project homepage](https://paoshou-china.github.io/VTIACG/) · [Browse source](https://github.com/PaoShou-China/VTIACG)

![Stylized 2D guidance simulation geometry](assets/hero.svg)

## Project overview / 项目概览

The repository contains separate script collections rather than one installable package or unified runner. The simulation code defines missile and virtual-target models; additional scripts run batches or parameter searches and produce comparisons and plots.

本仓库按编号目录保存若干独立的仿真脚本，并非统一安装包或单一运行器。仿真代码包含导弹和虚拟目标模型；其他脚本用于批量运行、参数搜索、结果比较和绘图。

## Repository map / 目录结构

| Path | Contents |
| --- | --- |
| 0/ | main.py: standalone missile/virtual-target simulation and plotting |
| 1/ | Simulation, batch, parameter search, comparison, plotting |
| 2/ | Simulation, batch, parameter/seed/state search, comparison, plotting |
| 3/ | Simulation, comparison, plotting |

Run scripts from their own directory. Files with the same name in different numbered folders are separate experiment variants.

## Requirements / 环境依赖

The inspected scripts use Python, NumPy, and Matplotlib. Some experiment scripts may require additional libraries; check their imports before running.

~~~bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy matplotlib
~~~

## Quick start / 快速开始

Run the standalone example from folder 0:

~~~bash
cd 0
python main.py
~~~

For another experiment, enter its numbered folder and run the desired script, for example python simulation.py or python batch.py. Inspect its main block first: batch or parameter-search scripts may run longer than a single simulation. Plot windows require a graphical environment.

## Reproduction notes / 复现说明

- Record the folder, script, parameters, random seed, and package versions with each result.
- Some scripts seed NumPy at module scope; others may use separate or no seed handling.
- There is no shared experiment configuration or single runner at present; use each script's current parameters.

复现实验时请记录目录、脚本、参数、随机种子及依赖版本。部分脚本在模块级设置 NumPy 随机种子，其他脚本可能使用不同策略。本仓库当前没有统一配置文件或总运行入口，应以所选脚本中的参数为准。
