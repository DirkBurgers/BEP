# Experiment Summary: Changing the parameterization of α, β₁, β₂

## Observations 
The only change in variation of ξ is the amplitude in the orientation plot. However, the behaviour in the orientation plot is still the same, but only the scaling is different. Another noteworthy observation was that, for each fixed gamma, the number of training steps was exactly the same for all ξ.

## Setting

### Data Parametrs 


| Parameter | Value                            |
| --------- | -------------------------------- |
| d         | 1                                |
| X         | [[-1/2], [-1/6], [1/6], [1/2]]   |
| Y         | [0.25, 0.03, 0.03, 0.25]         |


### Neural Network Parameters

| Parameter        | Value                |
| ---------------- | -------------------- |
| m                | 1000                 |
| γ                | {0.5, 1.0, 1.75}     |
| γ′               | 0.0                  |

### Training Parameters
| Parameter        | Value                |
| ---------------- | -------------------- |
| Optimizer        | Gradient descent     |
| Learning rate    | α / m                |