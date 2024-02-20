module MyTwoLayerNN

using Random, Distributions

export TwoLayerNN, TrainingData, train!, forward, summary

# Constants
const SEED = 123
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

# Risk function
Rs(x, y) = sum(z -> z^2, x - y) / (2 * length(x))
∂Rs(x, y) = sum(x - y) / length(x)

# Creates structure to store the NN data
struct TwoLayerNN{T<:Real, F, G}
    w::Matrix{T}          # Weights to hidden layer
    a::Vector{T}          # Weights from hidden layer
    b::Vector{T}          # Bias
    α::T                  # Scaling factor
    σ::ActivationFunction{F, G}   # Activation function
end
TwoLayerNN(d::T, m::T, α::Float64, β₁::Float64, β₂::Float64; σ=ReLu) where {T <: Integer} = begin
    # Set seed
    Random.seed!(SEED)

    # Initialize weights and biases
    w::Matrix{Float64} = rand(Normal(0, β₂), m, d)
    a::Vector{Float64} = rand(Normal(0, β₁), m)
    b::Vector{Float64} = rand(Normal(0, β₂), m)

    # Create the NN
    TwoLayerNN(w, a, b, α, σ)
end
TwoLayerNN(d::T, m::T, γ::Float64, γ′::Float64; σ=ReLu) where {T <: Integer} = begin
    # Parameters
    α = m^(γ - γ′)
    β₁ = m^(-γ′)
    β₂ = d^(-γ′)

    # α = 1.0
    # β₁ = m^(-(γ + γ′) / 2)
    # β₂ = m^(-(γ - γ′) / 2)

    # Create the NN
    TwoLayerNN(d, m, α, β₁, β₂; σ=σ)
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
function train!(nn::TwoLayerNN, trainData::TrainingData; debug=false)
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

    # Gradient descent
    for step = 1:steps
        # Reset gradiants
        fill!(gradData.∇a, 0.)
        fill!(gradData.∇w, 0.)
        fill!(gradData.∇b, 0.)

        # Sum the gradiant for all data points in the training data
        for (x, y) ∈ zip(trainData.x, trainData.y)
            updateGradiant!(gradData, nn, x, y)
        end

        # Apply the optimizer
        applyOptimizer!(optimizer, nn, gradData)

        # TEST: print accuracy
        current_risk = Rs(forward(nn, trainData.x), trainData.y)
        if debug
            println("Risk = ", current_risk)
        end
        if current_risk < TOLARANCY
            println("Number of steps: ", step)
            break
        end
        
    end
end

# Calculates the ∇ of the NN with the ith data point and adds it to the total ∇ 
function updateGradiant!(grads, nn::TwoLayerNN, x, y)
    # Aliases 
    inHL = grads.inL
    outHL = grads.outL
    ∇a = grads.∇a
    ∇w = grads.∇w
    ∇b = grads.∇b
    
    a = nn.a
    α = nn.α

    # Calculate the prediction of ith data point and store 
    # the values that went in and out the hidden layer
    predicted = forward!(nn, x, inHL, outHL)

    # Calculate the gradiants
    ∂Risk∂p = (predicted - y) / (α * length(x))

    # Update ∇
    @. ∇a += ∂Risk∂p * outHL
    @. ∇w += ∂Risk∂p * a * nn.σ.∂σ.(inHL) * x'
    @. ∇b += ∂Risk∂p * a * nn.σ.∂σ.(inHL)
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
end

end;