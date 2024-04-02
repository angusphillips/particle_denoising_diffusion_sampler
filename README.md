# Particle Denoising Diffusion Sampler

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

Simple installation instructions are:

```
git clone https://github.com/angusphillips/pdds.git
cd pdds
conda create -n {ENV_NAME} python=3.9
source activate {ENV_NAME}
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
pip install -e .
```

You can test the installation by running our unit tests:
```
python -m unittest discover tests
```

## Code Layout

We use Hydra to manage configs and WandB for logging. Default configs for all experiments in the paper are found in `/config/`. 

An interactive demonstration of our code can be found in `/notebooks/gaussian_example.ipynb`.

Our experiments can be run from the command line: 
```
python main.py experiment=<EXPERIMENT> <additional_arg=additional_value>
```
For example to run our method on the 2d Gaussian mixture task using 4 steps:
```
python main.py experiment=difficult_2d steps_mult=4 seed=1
```


The main arguments of interest are:
* `experiment`: specifies the task, available options in `/config/experiments/`
* `steps_mult`: controls the number of steps (resolves as base_steps * steps_mult)
* `seed`: sets global seed

Further arguments can be explored in the config files. 

## Cite
If you use this code in your work, please cite our paper:
```
@misc{phillips2024particle,
      title={Particle Denoising Diffusion Sampler}, 
      author={Angus Phillips and Hai-Dang Dau and Michael John Hutchinson and Valentin De Bortoli and George Deligiannidis and Arnaud Doucet},
      year={2024},
      eprint={2402.06320},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
