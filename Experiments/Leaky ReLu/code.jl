using Makie, CairoMakie

include("../../MyTwoLayerNN/MyTwoLayerNN.jl")
include("../experimentplots.jl")

FILE_PATH = dirname(@__FILE__) * "\\"

# Data parameters
d = 1
dataX = [[-1/2], [-1/6], [1/6], [1/2]]
dataY = [0.25, 0.03, 0.03, 0.25]

# NN parameters
m = 10_000
γ = 1.0
γ′ = 0.0

# Training parameters
steps = 100_000_000
learning_rate = 10.0
myTrainingData = MyTwoLayerNN.TrainingData(dataX, dataY, learning_rate, steps)

for leaky_relu_const ∈ 0.99 : 0.1 : 0.99
    # Reinstantiate the NN
    myNN = MyTwoLayerNN.TwoLayerNN(d, m, γ, γ′; σ=MyTwoLayerNN.LeakyReLU(leaky_relu_const))
    
    # Create copy of initial weights and biases
    initParms = (a = copy(myNN.a), w = copy(myNN.w), b = copy(myNN.b))

    # Train the NN 
    MyTwoLayerNN.train!(myNN, myTrainingData; debug = true)

    # Create and save the plots
    f = Figure()
    ax = Axis(f[1, 1], xticks=-0.5:0.25:0.5) 
    createlineplot!(ax, myNN, myTrainingData)
    # f |> display
    save(FILE_PATH * "line c=$leaky_relu_const.png", f)

    f = createorientiationplot(initParms, myNN)
    # f |> display
    save(FILE_PATH * "orientation c=$leaky_relu_const.png", f)
    
    println("Saved plot with γ=$γ, γ'=$γ′, leaky_relu_const=$leaky_relu_const.")
end