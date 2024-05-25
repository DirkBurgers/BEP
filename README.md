# BEP

### Setup environment
First the environment has to be activated using Pkg. For this 

- Run `activate path_to_project` with Pkg, where `path_to_project` is the path to the root folder.
- Run `instantiate` with Pkg.

Now everything should be set up.

### Structure

- `data/` contains data of NN's
- `src/` contains the code for the neural network
- `3dplotapp.jl` to run a 3D visualization of the position of neurons over time
- `test.jl` to train a test neural network
- `datageneration.jl` to create and save the initial and trained NN, but not intermediate results.
- `trainingdatageneration.jl` to create and save data of the training process of an NN.