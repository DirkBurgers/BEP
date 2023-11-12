using Flux, Statistics, Plots, Random, Distributions

# NN parameters
d = 1       # Dimension data
m = 1000    # Nuber of hidden neurons
γ = 1

# Calculates quantities
α = m^γ
β₁ = 1
β₂ = 1

# Training data 
n = 4
dataX = [-1/2, -1/3, 1/3, 1/2]'
dataY = [0.25, 0.03, 0.03, 0.25]'
learning_rate = 0.05                 # for α = √m -> 1.5

# Create model
model = Chain(
    Dense(rand(Normal(0, β₂), m, 1), rand(Normal(0, β₂), m), relu),
    Dense(rand(Normal(0, β₁), 1, m), false, z -> z / α)
)

loss(x, y) = Flux.mse(model(x), y) / 2

loader = Flux.DataLoader((dataX, dataY), batchsize=4, shuffle=true);
ps = Flux.params(model)
opt = Adam(learning_rate)

# Train model
loss_history = []

epochs = 100_000

for epoch in 1:epochs
    Flux.train!(loss, ps, loader, opt)
    # train_loss = loss(dataX, dataY)
    # push!(loss_history, train_loss)
    # println("Epoch = $epoch : Training loss = $train_loss")
end


# Create picture
gridPlot = (-0.5:0.1:0.5 .|> identity)'
Plots.plot(gridPlot', model(gridPlot)')
Plots.scatter!(dataX, dataY)