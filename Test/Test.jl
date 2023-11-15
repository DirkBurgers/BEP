module Test

include("../MyTwoLayerNN/MyTwoLayerNN.jl")
include("../MyTwoLayerNN/plotsNN.jl")

function main()
    # Create the NN
    d = 1       # Dimension data
    m = 1000    # Nuber of hidden neurons
    γ = 1
    γ′ = 0
    myNN = MyTwoLayerNN.TwoLayerNN(d, m, γ, γ′)

    # Create training data 
    n = 4
    dataX = [[-1/2], [-1/3], [1/3], [1/2]]
    dataY = [0.25, 0.03, 0.03, 0.25]
    # dataX = [[-1/2, -1/2], [-1/3, -1/3], [1/3, 1/3], [1/2, 1/2]]
    # dataY = [0.25, 0.03, 0.03, 0.25]

    learning_rate = 0.01
    steps::Int32 = 10_000
    myTrainingData = MyTwoLayerNN.TrainingData(n, dataX, dataY, learning_rate, steps)

    # Create copy of initial weights and biases
    initParms = (a = copy(myNN.a), w = copy(myNN.w), b = copy(myNN.b))

    # Train
    @time MyTwoLayerNN.train!(myNN, myTrainingData)
    #@benchmark trainNN!($myNN, $steps)

    # View result
    MyTwoLayerNN.summary(myNN)
    plotNN(myNN, dataX, dataY, -0.5:0.1:0.5)
    orientationPlot(initParms, myNN)
end;

main()

end;