using Makie, CairoMakie

include("../../MyTwoLayerNN/MyTwoLayerNN.jl")
include("../experimentplots.jl")

FILE_PATH = dirname(@__FILE__) * "\\"
FILE_EXTENSION = ".pdf"

# Data parameters
d = 1
dataX = [[-1/2], [-1/6], [1/6], [1/2]]
dataY = [0.25, 0.03, 0.03, 0.25]

# NN parameters
m = 1000
γ = 1.75
γ′ = 0.0

# Training parameters
steps = 10_000_000
learning_rate = 1000.0
myTrainingData = MyTwoLayerNN.TrainingData(dataX, dataY, learning_rate, steps)

# for leaky_relu_const ∈ [0.95] #0.0 : 0.1 : 0.9
#     # Reinstantiate the NN
#     myNN = MyTwoLayerNN.TwoLayerNN(d, m, γ, γ′; σ=MyTwoLayerNN.LeakyReLU(leaky_relu_const))
    
#     # Create copy of initial weights and biases
#     initParms = (a = copy(myNN.a), w = copy(myNN.w), b = copy(myNN.b))

#     # Train the NN 
#     MyTwoLayerNN.train!(myNN, myTrainingData; debug=true)

#     # Create and save the plots
#     f = Figure()
#     ax = Axis(f[1, 1], xticks=-0.5:0.25:0.5) 
#     createlineplot!(ax, myNN, myTrainingData)
#     f |> display
#     # save(FILE_PATH * "line c=$leaky_relu_const.png", f)

#     f = createorientiationplot(initParms, myNN)
#     f |> display
#     # save(FILE_PATH * "orientation c=$leaky_relu_const.png", f)
    
#     println("Saved plot with γ=$γ, γ'=$γ′, leaky_relu_const=$leaky_relu_const.")
# end

function simulationstep(leaky_relu_const)
    # Reinstantiate the NN
    myNN = MyTwoLayerNN.TwoLayerNN(d, m, γ, γ′; σ=MyTwoLayerNN.LeakyReLU(leaky_relu_const))
    
    # Create copy of initial weights and biases
    initParms = (a = copy(myNN.a), w = copy(myNN.w), b = copy(myNN.b))

    # Train the NN 
    MyTwoLayerNN.train!(myNN, myTrainingData; debug=true)
    
    return myTrainingData, initParms, myNN
end

function create_combined_image()
    f = Figure(size = (1200, 600))
    for (i, c) ∈ enumerate([0.0, 0.5, 0.95])
        # Run simulation step
        myTrainingData, initParms, myNN = simulationstep(c)

        # Create and save the plots
        ax = Axis(f[1, i], xticks=-0.5:0.25:0.5, xlabel="x", ylabel="y") 
        createlineplot!(ax, myNN, myTrainingData)

        ax = Axis(f[2, i]) 
        createorientiationplot!(ax, initParms, myNN)
        
        println("Saved plot with γ=$γ, γ'=$γ′, c=$c.")
    end

    rowgap!(f.layout, 30)

    Label(f[1, 1, Bottom()], L"\text{(a) } c=0.0", valign=-1, padding=(0, 0, 5, 0))
    Label(f[1, 2, Bottom()], L"\text{(b) } c=0.5", valign=-1, padding=(0, 0, 5, 0))
    Label(f[1, 3, Bottom()], L"\text{(c) } c=0.95", valign=-1, padding=(0, 0, 5, 0))
    Label(f[2, 1, Bottom()], L"\text{(d) } c=0.0", valign=-1, padding=(0, 0, 5, 50))
    Label(f[2, 2, Bottom()], L"\text{(e) } c=0.5", valign=-1, padding=(0, 0, 5, 50))
    Label(f[2, 3, Bottom()], L"\text{(f) } c=0.95", valign=-1, padding=(0, 0, 5, 50))

    save(FILE_PATH * "combined" * FILE_EXTENSION, f)
end

create_combined_image()