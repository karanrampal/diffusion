![Diffusion](https://github.com/karanrampal/diffusion/actions/workflows/main.yml/badge.svg)

# Diffusion
DDPM diffusion from scratch using Accelerator from HuggingFace to train models on multiple GPU's, nodes, precisions, FSDP support etc.

![Sprite gif](assets/sprite_diffusion.gif)

## Install
If you are using conda, you can do,
```
make install
```

If you are not using conda, then just do the following from your virtual environment,
```
make install_ci
```

## Train
To just run training as it is without acceleration run the following command from projects root,
```
python src/run.py
```

To run training with acceleration i.e. on multiple GPU's, Mixed precision etc., run the following,
```
accelerate config
```
and select the necessary config options, then run
```
accelerate launch src/run.py
```

## Visualize
To run inference you can use the `notebooks/visualize.ipynb` to run it and check the results
