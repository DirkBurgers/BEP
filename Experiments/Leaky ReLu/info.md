# Experiment Summary: Varying Leaky ReLU Constant

## Observations 

The orientation plots show that most of the "pillars" in the data points are aligned with orientations of 0 and π. However, it's important to note that this is a specific case and may not always be true for different datasets. Additionally, as the Leaky ReLU constant approaches 1, the "pillars" become more prominent and their amplitudes increase.

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
| Learning rate    | {0.1, 10.0, 1000.0}  |

Where the learning rate and gamma are used correspondingly, i.e. γ=0.5 with learning rate 0.1 etc.