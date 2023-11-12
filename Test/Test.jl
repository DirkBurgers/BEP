module Test

include("../MyTwoLayerNN/MyTwoLayerNN.jl")

function summaryNN(nn::MyTwoLayerNN.TwoLayerNN)
    println("d = $(nn.d), m = $(nn.m), α = $(nn.α)")
    
    w_max = max(nn.w ...)
    w_min = min(nn.w ...)
    w_avg = sum(nn.w) / length(nn.w)
    println("w: max = $w_max, min = $w_min, avg = $w_avg")

    a_max = max(nn.a ...)
    a_min = min(nn.a ...)
    a_avg = sum(nn.a) / length(nn.a)
    println("a: max = $a_max, min = $a_min, avg = $a_avg")

    b_max = max(nn.bias ...)
    b_min = min(nn.bias ...)
    b_avg = sum(nn.bias) / length(nn.bias)
    println("Bias: max = $b_max, min = $b_min, avg = $b_avg")
end

function main()
    # Create the NN
    d = 1     # Dimension data
    m = 1000    # Nuber of hidden neurons
    γ = 1
    myNN = MyTwoLayerNN.TwoLayerNN(d, m, γ)

    # Create training data 
    n = 4
    dataX = [-1/2, -1/3, 1/3, 1/2]
    dataY = [0.25, 0.03, 0.03, 0.25]
    learning_rate = 1                 # for α = √m -> 1.5
    steps::Int32 = 10_000
    myTrainingData = MyTwoLayerNN.TrainingData(n, dataX, dataY, learning_rate, steps)

    # Train
    @time MyTwoLayerNN.trainNN!(myNN, myTrainingData)
    #@benchmark trainNN!($myNN, $steps)

    # View result
    #summaryNN(myNN)
    MyTwoLayerNN.plotNN(myNN, dataX, dataY, -0.5:0.1:0.5)
end;

main()

end;