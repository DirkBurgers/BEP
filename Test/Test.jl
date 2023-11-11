module Test

include("../MyTwoLayerNN/MyTwoLayerNN.jl")

function main()
    # Create the NN
    d = 1     # Dimension data
    m = 32    # Nuber of hidden neurons
    myNN = MyTwoLayerNN.TwoLayerNN(d, m)

    # Create training data 
    n = 4
    dataX = [-1/2, -1/3, 1/3, 1/2]
    dataY = [0.25, 0.03, 0.03, 0.25]
    learning_rate = 0.01
    steps::Int32 = 100_000
    myTrainingData = MyTwoLayerNN.TrainingData(n, dataX, dataY, learning_rate, steps)

    # Train
    @time MyTwoLayerNN.trainNN!(myNN, myTrainingData)
    #@benchmark trainNN!($myNN, $steps)

    # View result
    MyTwoLayerNN.plotNN(myNN, dataX, dataY, -0.5:0.1:0.5)
end;

main()

end;