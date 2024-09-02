## Configuration File Overview

The configuration files are used to customize the model hyperparameters. We describe the parameters used to alter the EDL.

### EDL Loss Parameters

All of these are under the `loss` parameter in the config file. 

| **Parameter**        | **Type**   | **Description**                                                                                                  |
|----------------------|------------|------------------------------------------------------------------------------------------------------------------|
| `loss_fn`            | String     | The loss function chosen from [`nll`, `mse`] for EDL and non-EDL models respectively.                            |
| `edl`                | Boolean    | Enables or disables outputting EDL parameters.                                                                   |
| `edl_act`            | String     | Output activation function chosen from [`softplus`, `relu`] for EDL and non-EDL models respectively.             |
| `lambda`             | Float      | The strength of the new evidential regularizer. Higher `lambda` results in better uncertainties.                 |
| `lambda_increasing`  | Boolean    | Enables or disables whether to slowly increase `lambda` by `slope` each epoch starting from 0.                   |
| `slope`              | Float      | The slope of how sharply to increase `lambda` if `lambda_increasing` is True.                                    |
| `kl`                 | Boolean    | Enables or disables using KL-Divergence regularizer as opposed to the new regularizer.                           |
| `omega`              | Float      | Hyperparameter for KL-Divergence regularizer.                                                                    |
| `late_start`         | Integer    | Determines at what epoch to include the new evidential regularizer in the loss function.                         |


### Example Configuration

```yaml
loss:
  loss_fn: 'nll'
  edl: True
  edl_act: 'softplus'
  lambda: 1
  lambda_increasing: True
  slope: 2e-1
  kl: False
  omega: 0.01
  late_start: 0
```
