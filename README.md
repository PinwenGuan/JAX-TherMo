# JAX-TherMo
A demo code for implementation of differentiable thermodynamic modeling in [JAX](https://github.com/google/jax).

## Install 

No installation is needed. Just download the files in one folder.<br>

[JAX](https://github.com/google/jax) should be installed first.<br>

## Results

After running the code, you should be able to get the results as in the "Results" folder, including the loss function and its decomposition into different contributions, the gradient of the loss function, the model parameters and the Gibbs energies of involved phases, all evolving with the training process, as well as the predicted phase diagram based on the trained model.<br> 

## Reference

Please cite the reference below if you use this code in your work:<br>

Guan, Pin-Wen. "Differentiable thermodynamic modeling." arXiv preprint arXiv:2102.10705 (2021).<br>

```
@article{guan2021differentiable,
  title={Differentiable thermodynamic modeling},
  author={Guan, Pin-Wen},
  journal={arXiv preprint arXiv:2102.10705},
  year={2021}
}
```
