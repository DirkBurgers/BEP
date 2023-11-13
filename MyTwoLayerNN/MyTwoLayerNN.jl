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
TwoLayerNN(d::T, m::T, γ::F, γ′::F) where {T <: Integer, F <: Real} = begin
    # Set seed
    Random.seed!(123)

    # Parameters
    α = m^(γ + γ′)
    β₁= m^(-γ′)
    β₂= d^(-γ′)

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
    m = nn.m

    # Create aliases for Training data 
    n = trainData.n
    learning_rate = trainData.learning_rate
    steps = trainData.steps

    # Allocate memory for gradiants 
    myTempData = TempTrainingData(m)
    ∇a = myTempData.∇a
    ∇w = myTempData.∇w
    ∇b = myTempData.∇b

    # Allocate memory for Adam
    adamParms = (   ma = zeros(m), mw = zeros(m), mb = zeros(m), 
                    va = zeros(m), vw = zeros(m), vb = zeros(m), 
                    βₘ = 0.9, βᵥ = 0.999, βₚ = [0.9, 0.999], η = learning_rate
                )

    # TODO: Remove later 
    # checksEvery::Int32 = 100
    # errors = [Rs(forward(nn, dataX), dataY)]

    for s = 1:steps
        # Reset gradiants
        ∇a .= 0
        ∇w .= 0
        ∇b .= 0

        # Sum the gradiant for all data points
        for i = 1:n
            updateGradiant!(i, nn, trainData, myTempData)
        end

        # Aplly adam
        applyAdam!(adamParms, nn, myTempData)

        # TODO remove later 
        # if (s % checksEvery == 0)
        #     push!(errors, Rs(forward(nn, dataX), dataY))
        # end
    end

    # TODO remove later
    # (display ∘ plot)(0:checksEvery:steps, errors .|> log10)
    # println("Last risk: $(errors[end])")
end

function applyAdam!(o, nn::TwoLayerNN, temp::TempTrainingData)
    applyAdam!(o.ma, o.va, o.βₘ, o.βᵥ, o.βₚ, o.η, temp.∇a, nn.a)
    applyAdam!(o.mw, o.vw, o.βₘ, o.βᵥ, o.βₚ, o.η, temp.∇w, nn.w)
    applyAdam!(o.mb, o.vb, o.βₘ, o.βᵥ, o.βₚ, o.η, temp.∇b, nn.bias)

    o.βₚ[1] *= o.βₘ
    o.βₚ[2] *= o.βᵥ
end

function applyAdam!(m, v, βₘ :: Real, βᵥ :: Real, βₚ, η :: Real, ∇, w)
    @. m = βₘ * m + (1 - βₘ) * ∇
    @. v = βᵥ * v + (1 - βᵥ) * ∇ * ∇
    @fastmath @. w -=  η * m / (1 - βₚ[1]) / (√(v / (1 - βₚ[2])) + 1e-8)
end

function updateGradiant!(i::Int64, nn::TwoLayerNN, trainData::TrainingData, temp::TempTrainingData)
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