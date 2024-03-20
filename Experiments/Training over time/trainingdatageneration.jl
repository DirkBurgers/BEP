using LinearAlgebra
using JLD2

# Load MyTwoLayerNN if not loaded yet
(@isdefined MyTwoLayerNN) == false ?
begin
    include("../../MyTwoLayerNN/MyTwoLayerNN.jl")
    using .MyTwoLayerNN
    println("Included MyTwoLayerNN")
end : nothing

let
# Data parameters
d = 1
dataX = [[-1/2], [-1/6], [1/6], [1/2]]
dataY = [1/4, 1/30, 1/30, 1/4]

# NN parameters
m = 1_000
γ = 1.5
γ′ = -0.5

nn = TwoLayerNN(d, m, γ, γ′)

# Training parameters
learning_rate = 1000.0
max_steps = 10_000

training_data = TrainingData(dataX, dataY, learning_rate, max_steps)

# Create callback
t_data = [0.0]
nn_data = [copy(nn)]
function savecallback(nn, step, loss)
    if step % 100 == 0
        push!(t_data, step * learning_rate)
        push!(nn_data, copy(nn))
    end
end

# Train NN
train!(nn, training_data; callback=savecallback)

# Save data
DATA_FOLDER = joinpath(@__DIR__, "Training data")
jldsave(joinpath(DATA_FOLDER, "begin a-lag.jld2"); t_data=t_data, nn_data=nn_data, training_data=training_data) 

return nothing
end