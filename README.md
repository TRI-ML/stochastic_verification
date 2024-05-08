## Overview
This repository accompanies the paper **How Generalizable Is My Behavior Cloning Policy? A Statistical Approach to Trustworthy Performance Evaluation** submitted to IEEE RA-L. The purposes of this repository are to

- store our experimental data,
- show how we process our experimental data to produce the tables and figures in our paper,
- allow users to run their own experiments and statistical evaluations with diffusion policies in the robosuite simulator.

## Setup
Following the instructions in `README_diffusion_poliy.md`, 

> To reproduce our simulation benchmark results, install our conda environment on a Linux machine with Nvidia GPU. On Ubuntu 20.04 you need to install the following apt packages for mujoco:
>```console
>$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
>```
>We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
>```console
>$ mamba env create -f conda_environment.yaml
>```
>but you can use conda as well: 
>```console
>$ conda env create -f conda_environment.yaml
>```
>The `conda_environment_macos.yaml` file is only for development on MacOS and does not have full support for benchmarks.


## Analyzing Our Data
#### Simulation Experiments
The data from the simulation experiments is in the `results/` directory. 
The notebook `analyze_results_sim.ipynb` loads this data and produces confidence bounds. This notebook produces Table 1 and Figure 6 in the paper.

#### Hardware Experiment
The data from the hardware experiments is in the `results/pour_ice/` directory. 
The notebook `analyze_results_hardware.ipynb` loads this data and produces confidence bounds. The results are used to make Figure 7 in the paper.

#### Policy Comparison Experiment
**TODO**


## Analyzing The Bounds
Comparing the tightness of the binomial bound we use to the Clopper-Pearson bound is done in the [`binomial_CIs`](https://github.com/TRI-ML/binomial_CIs) repository. Specifically, the notebook `tradeoff_table.ipynb` produces Figure 3 in the paper.

Comparing the tightness of the CDF bound we use (based on the Kolmogorov-Smirnov test) to CDF bound from the DKW inequality is done in the **TODO** notebook. This notebook produces Figure 4 in the paper.

## Running New Experiments
To run new experiments you must first download the trained policies from the [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) paper. These policies can be found at the url [https://diffusion-policy.cs.columbia.edu/data/](https://diffusion-policy.cs.columbia.edu/data/).

At this url, the policies we evaluate in our paper are those given by the filepaths: 
```experiments/image/task/_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt```

where `task` is one of `can`, `lift`, `square`, `tool_hang`, `transport`.

Once the desired policy is downloaded, you can use `eval.py` to evaluate the policy in simulation.
In this file you can specify the policy, task, whether domain modification (i.e. OOD modification) is on, and how many policy rollouts to run. The data from the experiment will then be stored in a timestamped folder in the `results/` directory.



## Constructing Confidence Bounds
#### Lower Bounds on Success Probability
Find a lower bound on some unknown success probability $p$ given some observed successes and failures:
```
from binomial_cis import binom_ci

k = 5 # number of successes
n = 10 # number of trials
alpha = 0.05 # miscoverage probability

lb = binom_ci(k, n, alpha, 'lb')

```

#### Upper Bounds on CDF of Reward
Find an upper bound on some unknown CDF of reward $F(r)$ given some reward samples:
```
n = 10 # number of trials
alpha = 0.05 # miscoverage probability

F_ub = F_hat + KS_epsilon(n, alpha)

```
where `F_hat` is the empirical CDF, and `KS_epsilon` is the offset chosen to meet the coverage guarantee:

```
from scipy.special import smirnov

def KS_dist(n, epsilon):
    cdf_val = 1 - smirnov(n, epsilon)
    return np.clip(cdf_val, 0, 1)

def KS_epsilon(n, alpha, tol=1e-8):
    # find smallest epsilon such that KS_dist(n, alpha, epsilon) >= 1-alpha
    lb = 0
    ub = 1
    for i in range(100):
        epsilon = (ub - lb) / 2 + lb
        coverage = KS_dist(n, epsilon)
        if coverage >= 1-alpha:
            if coverage - (1-alpha) <= tol:
                return epsilon
            else:
                ub = epsilon
        else:
            lb = epsilon
    raise ValueError("Too few iterations on bisection search!")

```



## Changes to the Diffusion Policy Repository
This repository is a fork of the [diffusion policy repository](https://github.com/real-stanford/diffusion_policy). To conduct our research we made the following changes
- Modified `eval.py`.
- Modified `robomimic_image_runner.py`.
- Added `domain_alteration_wrapper.py` to robosuite package. Specifically, added the file.
`mambaforge/envs/stochastic_verification/lib/python3.9/site-packages/robosuite/wrappers/domain_modification_wrapper.py`.
- Added the `results/` directory. This holds the results for sim runs.
