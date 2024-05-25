using BenchmarkTools, JLD2

using MyTwoLayerNN


function main()
    # Create the NN
    d = 1        # Dimension data
    m = 1_000    # Number of hidden neurons
    γ = 1.5
    γ′ = -0.5
    myNN = TwoLayerNN(d, m, γ, γ′; σ=LeakyReLU(0.95))

    # Create training data
    dataX = [[-1/2], [-1/6], [1/6], [1/2]]
    dataY = [1/4, 1/30, 1/30, 1/4]

    # Training data
    learning_rate = 100_000.0 # 0.01 for γ 0.5, 1.0 and 1.0 for γ = 1.75
    steps = 1_000_000
    myTrainingData = TrainingData(dataX, dataY, learning_rate, steps)

    # Train
    @time train!(myNN, myTrainingData; debug=true)

    # Accuracy of network 
    println("Predicted value: ", forward(myNN, dataX))

    # View result
    summary(myNN)
end;

main()