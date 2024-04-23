using JLD2

include("../../MyTwoLayerNN/MyTwoLayerNN.jl")

# Create DATA_FOLDER
DATA_FOLDER = joinpath(@__DIR__, "leaky relu")
if !isdir(DATA_FOLDER)
    mkdir(DATA_FOLDER)
end

# Data parameters
d = 1
dataX = [[-1/2], [-1/6], [1/6], [1/2]]
dataY = [1/4, 1/30, 1/30, 1/4]

# NN parameters
m = 1_000
γ = 1.5
γ′ = -0.5

# Training parameters
learning_rate = 1000.0
max_steps = 100_000_000

training_data = MyTwoLayerNN.TrainingData(dataX, dataY, learning_rate, max_steps)

function simulationstep(training_data, leaky_constant)
    # Create the NN
    nn = MyTwoLayerNN.TwoLayerNN(d, m, γ, γ′; σ=MyTwoLayerNN.LeakyReLU(leaky_constant))

    # Create copy of inital weights 
    initialNN = MyTwoLayerNN.copy(nn)

    # Train the NN
    MyTwoLayerNN.train!(nn, training_data; debug=true)

    jldsave(joinpath(DATA_FOLDER, "run c=$leaky_constant.jld2"); initNN=initialNN, trainedNN=nn, trainingData=training_data)    
end

for c ∈ [0.99]
    simulationstep(training_data, c)
end