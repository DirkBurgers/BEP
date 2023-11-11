module MyTwoLayerNN
export main

using Random, Distributions, BenchmarkTools
import Plots: plot, scatter!

# Constants
const n = 4     # Number of data points
const learning_rate = 0.01

# Data
const dataX = [-1/2, -1/3, 1/3, 1/2]
const dataY = [0.25, 0.03, 0.03, 0.25]

# Functions
σ(x) = max(0, x)
∂σ(x) = x < 0 ? 0 : 1
Rs(x, y) = sum(z -> z^2, x - y) / 2n
∂Rs(x, y) = sum(x - y) / n

# Create structure to store the data
struct TwoLayerNN
    d::Integer                  # Number of input nodes
    m::Integer                  # Number of nodes in the hidden layer
    w::Vector{Float32}          # Weights to hidden layer
    a::Vector{Float32}          # Weights from hidden layer
    bias::Vector{Float32}       # Bias
    α::Float32                  # Scaling factor
end
TwoLayerNN(d::T, m::T) where {T <: Integer} = begin
    # Set seed
    Random.seed!(123)

    # Parameters
    α = 1
    β₁= √(1 / m)    # TODO set this dynamicly
    β₂= √(1 / d)

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
    inHiddenLayer::Vector{Float32}
    outHiddenLayer::Vector{Float32}
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

function trainNN!(nn::TwoLayerNN, steps::Int32)
    # Create aliases
    d = nn.d 
    m = nn.m
    w = nn.w 
    a = nn.a 
    bias = nn.bias
    α = nn.α

    # Allocate memory for weights
    ∇a = zeros(m)
    ∇w = zeros(m)
    ∂bias = zeros(m)
    
    # Allocate memory for training data 
    inL1 = zeros(Float32, m)
    outL1 = zeros(Float32, m)

    for s = 1:steps
        # Reset gradiants
        ∇a .= 0
        ∇w .= 0
        ∂bias .= 0

        # Sum the gradiant for all data points
        for i = 1:n
            predicted = forwardTrain!(nn, dataX[i], inL1, outL1)

            ∂Risk∂p = (predicted - dataY[i])

            # Calculate ∇a
            ∇a .+= ∂Risk∂p .* outL1

            # Calculate ∇b
            ∇w .+= ∂Risk∂p .* dataX[i] .* a .* ∂σ.(inL1)

            # Calculate ∂bias
            ∂bias .+= ∂Risk∂p .* a .* ∂σ.(inL1)
        end
        ∇a .= ∇a ./ (α * n)
        ∇w .= ∇w ./ (α * n)
        ∂bias .= ∂bias ./ (α * n)

        # Update weights and bias
        a .-= learning_rate .* ∇a
        w .-= learning_rate .* ∇w
        bias .-= ∂bias
    end
end

# Create plot 
function plotNN(nn::TwoLayerNN, x, y, range)
    plot(range, forward(nn, range))
    display(scatter!(x, y))
end

# Main function
function main()
    d = 1     # Dimension data
    m = 32    # Nuber of hidden neurons

    # Create the NN
    myNN = TwoLayerNN(d, m)

    # Steps
    steps::Int32 = 100_000

    # Train
    @time trainNN!(myNN, steps)
    #@benchmark trainNN!($myNN, $steps)

    # View result
    plotNN(myNN, dataX, dataY, -0.5:0.1:0.5)
end;

main()

end;