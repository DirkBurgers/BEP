using LinearAlgebra

using .MyTwoLayerNN
using Plots

# Plot of Neural network
function plotNN(nn::MyTwoLayerNN.TwoLayerNN, x, y, range)
    scatter(x, y)
    plotNN(nn, range)
end
plotNN(nn::MyTwoLayerNN.TwoLayerNN, range) = display(plot!(range, MyTwoLayerNN.forward(nn, range)))

# Orientation plot
function orientationPlot(old, new)
    # Calculate amplitude
    ampOld = abs.(old.a) .* norm.(zip(old.w, old.b))
    ampNew = abs.(new.a) .* norm.(zip(new.w, new.b))

    # Calculate orientation
    oriOld = angle.(old.w .+ im .* old.b)
    oriNew = angle.(new.w .+ im .* new.b)

    # Create plot
    scatter(oriOld, ampOld, label = "Initial")
    scatter!(oriNew, ampNew, label = "Final")
    xlims!(-π, π)
    xlabel!("Orientation")
    ylabel!("Amplitude") |> display
end