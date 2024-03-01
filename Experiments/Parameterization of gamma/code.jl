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
m = 1_000
γ = 1.75
γ′ = 0.0

# Training parameters
steps = 1_000_000

function simulationstep(ξ; debug = false)
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
    MyTwoLayerNN.train!(myNN, myTrainingData; debug)

    return myTrainingData, initParms, myNN
end

function run_simulations()
    for ξ ∈ -1.0 : 0.25 : 1.0
        # Run simulation step
        myTrainingData, initParms, myNN = simulationstep(ξ)
    
        # Create and save the plots
        f = Figure()
        ax = Axis(f[1, 1], xticks=-0.5:0.25:0.5) 
        createlineplot!(ax, myNN, myTrainingData)
        # f |> display
        save(FILE_PATH * "line xi=$ξ" * FILE_EXTENSION, f)
    
        f = createorientiationplot(initParms, myNN)
        # f |> display
        save(FILE_PATH * "orientation xi=$ξ" * FILE_EXTENSION, f)
        
        println("Saved plot with γ=$γ, γ'=$γ′, ξ=$ξ.")
    end
end

function create_combined_image()
    f = Figure(size = (1200, 600))
    for (i, ξ) ∈ enumerate([-1.0, 0.0, 1.0])
        # Run simulation step
        myTrainingData, initParms, myNN = simulationstep(ξ)

        # Create and save the plots
        ax = Axis(f[1, i], xticks=-0.5:0.25:0.5, xlabel="x", ylabel="y") 
        createlineplot!(ax, myNN, myTrainingData)

        ax = Axis(f[2, i]) 
        createorientiationplot!(ax, initParms, myNN)
        
        println("Saved plot with γ=$γ, γ'=$γ′, ξ=$ξ.")
    end

    rowgap!(f.layout, 30)

    Label(f[1, 1, Bottom()], L"\text{(a) } \xi=-1.0", valign=-1, padding=(0, 0, 5, 0))
    Label(f[1, 2, Bottom()], L"\text{(b) } \xi=0.0", valign=-1, padding=(0, 0, 5, 0))
    Label(f[1, 3, Bottom()], L"\text{(c) } \xi=1.0", valign=-1, padding=(0, 0, 5, 0))
    Label(f[2, 1, Bottom()], L"\text{(d) } \xi=-1.0", valign=-1, padding=(0, 0, 5, 50))
    Label(f[2, 2, Bottom()], L"\text{(e) } \xi=0.0", valign=-1, padding=(0, 0, 5, 50))
    Label(f[2, 3, Bottom()], L"\text{(f) } \xi=1.0", valign=-1, padding=(0, 0, 5, 50))

    save(FILE_PATH * "combined" * FILE_EXTENSION, f)
end

run_simulations()
# create_combined_image()