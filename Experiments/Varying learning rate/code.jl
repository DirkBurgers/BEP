using Makie, CairoMakie

include("../../MyTwoLayerNN/MyTwoLayerNN.jl")
include("../experimentplots.jl")

FILE_PATH = dirname(@__FILE__) * "\\"

# Data parameters
d = 1  # Dimension of the data
dataX = [[-1/2], [-1/6], [1/6], [1/2]]
dataY = [0.25, 0.03, 0.03, 0.25]

# NN parameters
m = 1_000
γ = 0.5
γ′ = 0.0

# Training parameters
steps = 100_000

for learning_rate ∈ 0.5:0.5:3.5
    # Reinstantiate the NN
    myNN = MyTwoLayerNN.TwoLayerNN(d, m, γ, γ′)
    
    # Reinstantiate the training data
    myTrainingData = MyTwoLayerNN.TrainingData(dataX, dataY, learning_rate, steps)
    
    # Create copy of initial weights and biases
    initParms = (a = copy(myNN.a), w = copy(myNN.w), b = copy(myNN.b))

    # Train the NN 
    MyTwoLayerNN.train!(myNN, myTrainingData)

    # Create and save the plots
    f = Figure()
    ax = Axis(f[1, 1], xticks=-0.5:0.25:0.5) 
    createlineplot!(ax, myNN, myTrainingData)
    # f |> display
    save(FILE_PATH * "line lr=$learning_rate.png", f)

    f = createorientiationplot(initParms, myNN)
    # f |> display
    save(FILE_PATH * "orientation lr=$learning_rate.png", f)

    
    println("Saved plot with ", learning_rate)
end