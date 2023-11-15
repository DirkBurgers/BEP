module MyTwoLayerNN

using Random, Distributions


# Activation functions
σ(x) = max(0, x)
∂σ(x) = x < 0 ? 0 : 1

# Risk function
Rs(x, y) = sum(z -> z^2, x - y) / (2 * length(x))
∂Rs(x, y) = sum(x - y) / length(x)

# Creates structure to store the NN data
struct TwoLayerNN
    d::Integer                  # Number of input nodes
    m::Integer                  # Number of nodes in the hidden layer
    w::Matrix{Float64}          # Weights to hidden layer
    a::Vector{Float64}          # Weights from hidden layer
    b::Vector{Float64}          # Bias
    α::Float64                  # Scaling factor
end
TwoLayerNN(d::T, m::T, γ, γ′) where {T <: Integer} = begin
    # Set seed
    Random.seed!(123)

    # Parameters
    α = m^(γ + γ′)
    β₁= m^(-γ′)
    β₂= d^(-γ′)

    # Initialize weights and biases
    w::Matrix{Float64} = rand(Normal(0, β₂), m, d)
    a::Vector{Float64} = rand(Normal(0, β₁), m)
    b::Vector{Float64} = rand(Normal(0, β₂), m)

    # Create the NN
    TwoLayerNN(d, m, w, a, b, α)
end 

# Structure to store the training data 
struct TrainingData 
    n::Int32
    x::Vector{Vector{Float64}}
    y::Vector{Float64}
    learning_rate::Float64
    steps::Int32
end

# Calculate output of NN
function forward(nn::TwoLayerNN, x::Vector{T}) where {T <: Real}
    nn.a' * (σ.(nn.w * x .+ nn.b)) ./ nn.α
end
forward(nn::TwoLayerNN, x::T) where {T <: Real} = forward(nn, [x])
forward(nn::TwoLayerNN, x::Vector{Vector{T}}) where {T <: Real} = map(z -> forward(nn, z), x)
forward(nn::TwoLayerNN, x::AbstractRange{T}) where {T <: Real} = map(z -> forward(nn, z), x)

function forward!(nn::TwoLayerNN, x, inHL::Vector{T}, outHL::Vector{T}) where {T <: Real}
    inHL .= nn.w * x .+ nn.b
    outHL .= σ.(inHL)
    return nn.a' * outHL ./ nn.α    
end

# Trains the NN with the training data
function train!(nn::TwoLayerNN, trainData::TrainingData)
    # Create aliases for data 
    d = nn.d
    m = nn.m
    n = trainData.n
    steps = trainData.steps

    # Allocate memory for gradiants and values that go in the hidden layer and out
    gradData = (
        ∇a = zeros(m), ∇w = zeros(m, d), ∇b = zeros(m), 
        inL = zeros(m), outL = zeros(m)
    )

    # Allocate memory for Adam optimization
    adamParms = (
        ma = zeros(m), mw = zeros(m, d), mb = zeros(m), 
        va = zeros(m), vw = zeros(m, d), vb = zeros(m), 
        βₘ = 0.9, βᵥ = 0.999, βₚ = [0.9, 0.999], η = trainData.learning_rate
    )

    # Gradient descent
    for s = 1:steps
        # Reset gradiants
        fill!(gradData.∇a, 0)
        fill!(gradData.∇w, 0)
        fill!(gradData.∇b, 0)

        # Sum the gradiant for all data points
        for i = 1:n
            updateGradiant!(i, nn, trainData, gradData)
        end

        # Apply adam
        applyAdam!(adamParms, nn, gradData)

    end
end

# Calculates the ∇ of the NN with the ith data point and adds it to the total ∇ 
function updateGradiant!(i, nn::TwoLayerNN, trainData::TrainingData, ∇data)
    # Aliases 
    a = nn.a
    α = nn.α
    dataX = trainData.x
    dataY = trainData.y
    n = trainData.n
    inHL = ∇data.inL
    outHL = ∇data.outL
    ∇a = ∇data.∇a
    ∇w = ∇data.∇w
    ∇b = ∇data.∇b

    # Calculate the prediction of ith data point and store 
    # the values that went in and out the hidden layer
    predicted = forward!(nn, dataX[i], inHL, outHL)

    # Calculate the gradiants
    ∂Risk∂p = (predicted - dataY[i]) / (α * n)

    # Calculate ∇a
    @. ∇a += ∂Risk∂p * outHL

    # Calculate ∇b
    @. ∇w += ∂Risk∂p * a * ∂σ.(inHL) * dataX[i]'

    # Calculate ∇b
    @. ∇b += ∂Risk∂p * a * ∂σ.(inHL)
end

# Optimization method - used to update the ∇
function applyAdam!(o, nn::TwoLayerNN, ∇data)
    applyAdam!(o.ma, o.va, o.βₘ, o.βᵥ, o.βₚ, o.η, ∇data.∇a, nn.a)
    applyAdam!(o.mw, o.vw, o.βₘ, o.βᵥ, o.βₚ, o.η, ∇data.∇w, nn.w)
    applyAdam!(o.mb, o.vb, o.βₘ, o.βᵥ, o.βₚ, o.η, ∇data.∇b, nn.b)

    o.βₚ[1] *= o.βₘ
    o.βₚ[2] *= o.βᵥ
end

function applyAdam!(m, v, βₘ :: Real, βᵥ :: Real, βₚ, η :: Real, ∇, w)
    @. m = βₘ * m + (1 - βₘ) * ∇
    @. v = βᵥ * v + (1 - βᵥ) * ∇ * ∇
    @fastmath @. w -=  η * m / (1 - βₚ[1]) / (√(v / (1 - βₚ[2])) + 1e-8)
end

# Creates a short summary of the NN
function summary(nn::TwoLayerNN)
    printstyled("Summary of neural network 🧠: \n", color = :blue)
    println("d = $(nn.d), m = $(nn.m), α = $(nn.α)")
    
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