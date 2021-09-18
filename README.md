# JAX-TherMo
A demo code for implementation of differentiable thermodynamic modeling in [JAX](https://github.com/google/jax), taking the Cu-Rh system as an example.

## Install 

No installation for this code is needed. Just download the files in one folder.<br>

[JAX](https://github.com/google/jax) should be installed first.<br>

## Run the code

Use train.py to train the model.<br>

Use pd.py to calculate the phase diagram based on the trained model.<br>

## Results

After running the code, you should be able to get the results as in the "Results" folder, including the loss function and its decomposition into different contributions, the gradient of the loss function, the model parameters and the Gibbs energies of involved phases, all evolving with the training process, as well as the predicted phase diagram based on the trained model.<br> 

## Reference

Please cite the reference below if you use this code in your work:<br>

Guan, Pin-Wen. "Differentiable thermodynamic modeling." arXiv preprint arXiv:2102.10705 (2021).<br>

```
@article{GUAN2022114217,
title = {Differentiable thermodynamic modeling},
journal = {Scripta Materialia},
volume = {207},
pages = {114217},
year = {2022},
issn = {1359-6462},
doi = {https://doi.org/10.1016/j.scriptamat.2021.114217},
url = {https://www.sciencedirect.com/science/article/pii/S1359646221004978},
author = {Pin-Wen Guan},
keywords = {Thermodynamic modeling, Differentiable programming, Machine learning},
}
```
