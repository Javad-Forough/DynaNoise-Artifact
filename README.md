# DynaNoise Artifact

This repository contains the artifact for the paper titled:

**DynaNoise: Mitigating Membership Inference Attacks by Dynamic Probabilistic Noise Injection**

## Setup

1. (Optional) Create and activate a virtual environment before installing:

```bash
python3 -m venv dyna_env
source dyna_env/bin/activate
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing Table 3 Results
To reproduce the main results reported in Table 3, run the following:

```bash
python3 main_cifar10.py
python3 main_imagenet10.py
python3 main_sst2.py
```

Each script will train or load the target model, apply DynaNoise and SELENA defenses, evaluate attack success rates, and print final results. Output metrics will be stored in `.csv` files for reference.


## Reproducing Figures 3, 4, and 5
To generate the parameter sensitivity plots for DynaNoise (base variance, lambda scale, and temperature), run:

```bash
python3 graph_cifar10.py
python3 graph_imagenet10.py
python3 graph_sst2.py
```
The generated figures will be saved in the `fig/` directory.