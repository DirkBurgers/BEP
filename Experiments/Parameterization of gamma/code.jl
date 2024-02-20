using Makie, CairoMakie

include("../../MyTwoLayerNN/MyTwoLayerNN.jl")
include("../experimentplots.jl")

FILE_PATH = dirname(@__FILE__) * "\\"

# Data parameters
d = 1
dataX = [[-1/2], [-1/6], [1/6], [1/2]]
dataY = [0.25, 0.03, 0.03, 0.25]

# NN parameters
m = 1_000
γ = 0.5
γ′ = 0.0

# Training parameters
steps = 400_000

for ξ ∈ -1.0 : 0.25 : 1.0
    # Calculate α, β₁, and β₂
    α = m^(γ - 2ξ - γ′)
    β₁ = m^(-ξ - γ′)
    β₂ = m^(-ξ)

    # Change learning rate depending on α and m
    learning_rate = α / m
    myTrainingData = MyTwoLayerNN.TrainingData(dataX, dataY, learning_rate, steps)

    # Reinstantiate the NN
    myNN = MyTwoLayerNN.TwoLayerNN(d, m, α, β₁, β₂)
    
    # Create copy of initial weights and biases
    initParms = (a = copy(myNN.a), w = copy(myNN.w), b = copy(myNN.b))

    # Train the NN 
    MyTwoLayerNN.train!(myNN, myTrainingData; debug=false)

    # Create and save the plots
    f = Figure()
    ax = Axis(f[1, 1], xticks=-0.5:0.25:0.5) 
    createlineplot!(ax, myNN, myTrainingData)
    # f |> display
    save(FILE_PATH * "line xi=$ξ.png", f)

    f = createorientiationplot(initParms, myNN)
    # f |> display
    save(FILE_PATH * "orientation xi=$ξ.png", f)
    
    println("Saved plot with γ=$γ, γ'=$γ′, ξ=$ξ.")
end