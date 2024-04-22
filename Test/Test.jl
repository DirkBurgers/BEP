using BenchmarkTools, JLD2

# include("../MyTwoLayerNN/MyTwoLayerNN.jl")
# include("../MyTwoLayerNN/plotsNN.jl")

import .MyTwoLayerNN

function main()
    # Create the NN
    d = 1       # Dimension data
    m = 1_000    # Number of hidden neurons
    γ = 1.5
    γ′ = -0.5
    myNN = MyTwoLayerNN.TwoLayerNN(d, m, γ, γ′; symmetric = true)

    # Create training data 
    n = 4
    dataX = [[-1/2], [-1/6], [1/6], [1/2]]
    dataY = [1/4, 1/30, 1/30, 1/4]
    # dataX = [[-1/2, -1/2], [-1/3, -1/3], [1/3, 1/3], [1/2, 1/2]]
    # dataY = [0.25, 0.03, 0.03, 0.25]

    learning_rate = 4000.0 # 0.01 for γ 0.5, 1.0 and 1.0 for γ = 1.75
    steps = 1_000_000
    myTrainingData = MyTwoLayerNN.TrainingData(dataX, dataY, learning_rate, steps)

    # Create copy of initial weights and biases
    initNN = MyTwoLayerNN.copy(myNN)
    # initParms = (a = copy(myNN.a), w = copy(myNN.w), b = copy(myNN.b))

    # Train
    @time MyTwoLayerNN.train!(myNN, myTrainingData; debug=true)
    # @benchmark MyTwoLayerNN.train!($myNN, $myTrainingData)
    #@benchmark trainNN!($myNN, $steps)

    # Accuracy of network 
    println("Predicted value: ", MyTwoLayerNN.forward(myNN, dataX))

    # View result
    MyTwoLayerNN.summary(myNN)
    # plotNN(myNN, dataX, dataY, -0.5:0.01:0.5)
    # orientationPlot(initNN, myNN)
end;

main()