module MyTwoLayerNN

import Base: copy
using Random, Distributions

export TwoLayerNN, TrainingData, train!, forward, summary

# Constants
SEED = 123
const TOLARANCY = 1e-6

# Activation functions
struct ActivationFunction{F, G}
    σ::F
    ∂σ::G
end

_ReLu(x) = max(zero(x), x)
_∂ReLu(x) = x < zero(x) ? zero(x) : one(x)
const ReLu = ActivationFunction(_ReLu, _∂ReLu)

_LeakyReLU(x, c) = x < zero(x) ? c * x : x
_∂LeakyReLU(x, c) = x < zero(x) ? c : one(x)
const LeakyReLU(c) = ActivationFunction(x -> _LeakyReLU(x, c), x -> _∂LeakyReLU(x, c))

_SmoothLeakyReLU(x, c) = (x + c*x + sqrt((x - c * x)^2 + 0.1^2)) / 2
_∂SmoothLeakyReLU(x, c) = (1 + c + (c - 1)^2 * x / sqrt((x - c*x)^2 + 0.1^2)) / 2
const SmoothLeakyReLU(c) = ActivationFunction(x -> _SmoothLeakyReLU(x, c), x -> _∂SmoothLeakyReLU(x, c))

_Identity(x) = x
_∂Identity(x) = one(x)
const Identity = ActivationFunction(_Identity, _∂Identity)

# Risk function
Rs(x, y) = sum(z -> z^2, x - y) / (2 * length(x))
∂Rs(x, y) = sum(x - y) / length(x)
∂Rs(x, y, n) = sum(x - y) / n

# Creates structure to store the NN data
struct TwoLayerNN{T<:Real, F, G}
    w::Matrix{T}          # Weights to hidden layer
    a::Vector{T}          # Weights from hidden layer
    b::Vector{T}          # Bias
    α::T                  # Scaling factor
    σ::ActivationFunction{F, G}   # Activation function
end
TwoLayerNN(d::T, m::T, α::Float64, β₁::Float64, β₂::Float64; σ=ReLu, symmetric=false) where {T <: Integer} = begin
    # Set seed
    Random.seed!(SEED)

    # Initialize weights and biases
    m_random = symmetric ? m ÷ 2 : m

    w::Matrix{Float64} = rand(Normal(0, β₂), m_random, d)
    a::Vector{Float64} = rand(Normal(0, β₁), m_random)
    b::Vector{Float64} = rand(Normal(0, β₂), m_random)

    if symmetric 
        w = [w; -w]
        a = [a; a]
        b = [b; b]
    end

    # Create the NN
    TwoLayerNN(w, a, b, α, σ)
end
TwoLayerNN(d::T, m::T, γ::Float64, γ′::Float64; σ=ReLu, symmetric=false) where {T <: Integer} = begin
    # Parameters
    α = m^(γ - γ′)
    β₁ = m^(-γ′)
    β₂ = d^(-γ′)

    # α = 1.0
    # β₁ = m^(-(γ + γ′) / 2)
    # β₂ = m^(-(γ - γ′) / 2)

    # Create the NN
    TwoLayerNN(d, m, α, β₁, β₂; σ=σ, symmetric=symmetric)
end
TwoLayerNN(d::T, m::T, γ, γ′; σ=ReLu) where {T <: Integer} = TwoLayerNN(d, m, convert(Float64, γ), convert(Float64, γ′), σ)

# Structure to store the training data 
struct TrainingData{T<:Real, S<:Integer}
    x::Vector{Vector{T}}
    y::Vector{T}
    learning_rate::T
    steps::S
end

# Include files 
include("optimizers.jl")

# Calculate output of NN
function forward(nn::TwoLayerNN, x::Vector{T}) where {T <: Real}
    nn.a' * (nn.σ.σ.(nn.w * x .+ nn.b)) ./ nn.α
end
forward(nn::TwoLayerNN, x::T) where {T <: Real} = forward(nn, [x])
forward(nn::TwoLayerNN, x::Vector{Vector{T}}) where {T <: Real} = map(z -> forward(nn, z), x)

function forward!(nn::TwoLayerNN, x, inHL::Vector{T}, outHL::Vector{T}) where {T <: Real}
    inHL .= nn.w * x .+ nn.b
    outHL .= nn.σ.σ.(inHL)
    return nn.a' * outHL ./ nn.α    
end

# Trains the NN with the training data
function train!(nn::TwoLayerNN, trainData::TrainingData; debug=false, callback=missing, callback2=missing, stop_tol=true)
    # Create aliases for data
    steps = trainData.steps

    # Allocate memory for gradiants and values that go in the hidden layer and out
    gradData = (
        ∇a = zero(nn.a), ∇w = zero(nn.w), ∇b = zero(nn.b), 
        inL = zero(nn.b), outL = zero(nn.a)
    )

    # Initialize optimizer
    optimizer = SGDOptimzer(nn, trainData.learning_rate)
    # optimizer = AdamOptimizer(nn, trainData.learning_rate)

    # Allocate memory for predictions
    predictions = similar(trainData.y)

    # Gradient descent
    for step = 1:steps
        # Reset gradiants
        fill!(gradData.∇a, 0.)
        fill!(gradData.∇w, 0.)
        fill!(gradData.∇b, 0.)

        # Sum the gradiant for all data points in the training data
        for (i, (x, y)) ∈ enumerate(zip(trainData.x, trainData.y))
            predicted = forward!(nn, x, gradData.inL, gradData.outL)
            updateGradiant!(gradData, nn, x, y, predicted, length(trainData.x))
            predictions[i] = predicted
        end

        # Apply the optimizer
        applyOptimizer!(optimizer, nn, gradData)
        
        # Calculate the risk
        current_risk = Rs(predictions, trainData.y)
        
        # Print the risk to be able to stop training with an intterupt
        if debug && step % 10_000 == 0
            println("Risk = ", current_risk)
        end

        # If callback function is given, call it
        if !ismissing(callback)
            callback(nn, step, current_risk)
        end

        # Newer version of callback function
        if !ismissing(callback2)
            stoptraining = callback2(nn, step, current_risk; ∇=gradData)
            stoptraining isa Bool && break
        end

        # If TOLARANCY is reached, stop
        if stop_tol && current_risk < TOLARANCY
            println("Number of steps: ", step)
            break
        end
        
    end
end

# Calculates the ∇ of the NN with the ith data point and adds it to the total ∇ 
function updateGradiant!(grads, nn::TwoLayerNN, x, y, predicted, datapoints)
    # Aliases 
    inHL = grads.inL
    outHL = grads.outL
    ∇a = grads.∇a
    ∇w = grads.∇w
    ∇b = grads.∇b
    
    a = nn.a
    α = nn.α

    # Calculate the gradiants
    ∂Risk∂p = ∂Rs(predicted, y, datapoints) / α

    # Update ∇
    @. ∇a += ∂Risk∂p * outHL
    @. ∇w += ∂Risk∂p * a * nn.σ.∂σ.(inHL) * x'
    @. ∇b += ∂Risk∂p * a * nn.σ.∂σ.(inHL)

    return nothing
end

# Create copy of MyTwoLayerNN
function copy(nn::TwoLayerNN)
    TwoLayerNN(copy(nn.w), copy(nn.a), copy(nn.b), nn.α, nn.σ)
end

# Creates a short summary of the NN
function summary(nn::TwoLayerNN)
    printstyled("Summary of neural network 🧠: \n", color = :blue)
    println("d = $(size(nn.w)[2]), m = $(size(nn.w)[1]), α = $(nn.α)")
    
    w_max = max(nn.w ...)
    w_min = min(nn.w ...)
    w_avg = sum(nn.w) / length(nn.w)
    println("w: max = $w_max, min = $w_min, avg = $w_avg")

    a_max = max(nn.a ...)
    a_min = min(nn.a ...)
    a_avg = sum(nn.a) / length(nn.a)
    println("a: max = $a_max, min = $a_min, avg = $a_avg")

    b_max = max(nn.b ...)
    b_min = min(nn.b ...)
    b_avg = sum(nn.b) / length(nn.b)
    println("Bias: max = $b_max, min = $b_min, avg = $b_avg")

    return nothing
end

end;