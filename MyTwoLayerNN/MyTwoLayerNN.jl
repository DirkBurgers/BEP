module MyTwoLayerNN


using Random, Distributions, BenchmarkTools
import Plots: plot, plot!, scatter

# Activation functions
σ(x) = max(0, x)
∂σ(x) = x < 0 ? 0 : 1

# Risk function
Rs(x, y) = sum(z -> z^2, x - y) / (2 * length(x))
∂Rs(x, y) = sum(x - y) / length(x)

# Create structure to store the data
struct TwoLayerNN
    d::Integer                  # Number of input nodes
    m::Integer                  # Number of nodes in the hidden layer
    w::Vector{Float32}          # Weights to hidden layer
    a::Vector{Float32}          # Weights from hidden layer
    bias::Vector{Float32}       # Bias
    α::Float32                  # Scaling factor
end
TwoLayerNN(d::T, m::T, γ::F) where {T <: Integer, F <: Real} = begin
    # Set seed
    Random.seed!(123)

    # Parameters
    α = m^γ
    β₁= 1
    β₂= 1

    # Initialize weights and biases
    w::Vector{Float32} = rand(Normal(0, β₂), m)
    a::Vector{Float32} = rand(Normal(0, β₁), m)
    bias::Vector{Float32} = rand(Normal(0, β₂), m)

    TwoLayerNN(d, m, w, a, bias, α)
end 

function printNN(nn::TwoLayerNN)
    print("Weight w: ")
    print(size(nn.w))
    print(" Weight a: ")
    print(size(nn.a))
    print(" Bias: ")
    print(size(nn.bias))
    print(" Type:")
    println(typeof(nn))
end

# Training data 
struct TrainingData 
    n::Int32
    x::Vector{Float32}
    y::Vector{Float32}
    learning_rate::Float32
    steps::Int32
end

struct  TempTrainingData
    ∇a :: Vector{Float32}
    ∇w :: Vector{Float32}
    ∇b :: Vector{Float32}
    inL :: Vector{Float32}
    outL :: Vector{Float32}
end
TempTrainingData(m::Int64) = begin
    ∇a = zeros(m)
    ∇w = zeros(m)
    ∇b = zeros(m)
    inL = zeros(Float32, m)
    outL = zeros(Float32, m)
    TempTrainingData(∇a, ∇w, ∇b, inL, outL)
end

# Calculate output of NN
function forward(nn::TwoLayerNN, x::Real)
    nn.a' * (σ.(nn.w * x + nn.bias)) / nn.α
end
forward(nn::TwoLayerNN, x) = map(z -> forward(nn, z), x)

function forwardTrain!(nn::TwoLayerNN, x, inL1::Vector{Float32}, outL1::Vector{Float32})
    inL1 .= nn.w * x + nn.bias
    outL1 .= σ.(inL1)
    return nn.a' * outL1 / nn.α    
end

function trainNN!(nn::TwoLayerNN, trainData::TrainingData)
    # Create aliases for NN
    d = nn.d 
    m = nn.m
    w = nn.w 
    a = nn.a 
    bias = nn.bias
    α = nn.α

    # Create aliases for Training data 
    n = trainData.n
    learning_rate = trainData.learning_rate
    steps = trainData.steps

    # Allocate memory for gradiants 
    myTempData = TempTrainingData(m)
    ∇a = myTempData.∇a
    ∇w = myTempData.∇w
    ∂bias = myTempData.∇b

    # Allocate memory for velocity
    vela = zeros(m)
    velw = zeros(m)
    velb = zeros(m)

    # TODO: Remove later 
    # checksEvery::Int32 = 100
    # errors = [Rs(forward(nn, dataX), dataY)]

    for s = 1:steps
        # Reset gradiants
        ∇a .= 0
        ∇w .= 0
        ∂bias .= 0

        # Sum the gradiant for all data points
        for i = 1:n
            gradiant!(i, nn, trainData, myTempData)
        end

        # Momentum
        @. vela = 0.9 * vela - learning_rate * ∇a
        @. velw = 0.9 * velw - learning_rate * ∇w
        @. velb = 0.9 * velb - learning_rate * ∂bias

        a .+= vela
        w .+= velw
        bias .+= velb

        # Update weights and bias
        # a .-= learning_rate .* ∇a
        # w .-= learning_rate .* ∇w
        # bias .-= ∂bias

        # TODO remove later 
        # if (s % checksEvery == 0)
        #     push!(errors, Rs(forward(nn, dataX), dataY))
        # end
    end

    # TODO remove later
    # (display ∘ plot)(0:checksEvery:steps, errors .|> log10)
    # println("Last risk: $(errors[end])")
end

function gradiant!(i::Int64, nn::TwoLayerNN, trainData::TrainingData, temp::TempTrainingData)
    # Aliases 
    a = nn.a
    α = nn.α
    dataX = trainData.x
    dataY = trainData.y
    n = trainData.n
    inL1 = temp.inL
    outL1 = temp.outL
    ∇a = temp.∇a
    ∇w = temp.∇w
    ∇b = temp.∇b

    predicted = forwardTrain!(nn, dataX[i], inL1, outL1)

    ∂Risk∂p = (predicted - dataY[i]) / (α * n)

    # Calculate ∇a
    @. ∇a += ∂Risk∂p * outL1

    # Calculate ∇b
    @. ∇w += ∂Risk∂p * a * ∂σ.(inL1) * dataX[i]

    # Calculate ∇b
    @. ∇b += ∂Risk∂p * a * ∂σ.(inL1)
end

# Create plot 
function plotNN(nn::TwoLayerNN, x, y, range)
    scatter(x, y)
    plotNN(nn, range)
end
plotNN(nn::TwoLayerNN, range) = display(plot!(range, forward(nn, range)))

end;