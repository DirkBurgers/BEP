using LinearAlgebra

using .MyTwoLayerNN
using Plots

# Plot of Neural network
function plotNN(nn::MyTwoLayerNN.TwoLayerNN, x, y, range)
    scatter(x .|> z -> z[1], y)
    plotNN(nn, range)
end
plotNN(nn::MyTwoLayerNN.TwoLayerNN, range) = plot!(range, [MyTwoLayerNN.forward(nn, x) for x in range]) |> display

# Orientation plot
function orientationPlot(old, new)
    # Calculate amplitude
    ampOld = abs.(old.a) .* norm.(eachrow([old.w old.b]))
    ampNew = abs.(new.a) .* norm.(eachrow([new.w new.b]))

    # Calculate orientation
    if size(old.w)[2] == 1
        oriOld = angle.(old.w .+ im .* old.b)
        oriNew = angle.(new.w .+ im .* new.b)
    else
        oriOld = angle.(sqrt.(norm.(eachrow([old.w old.b])).^2 .- (old.b .^2)) .+ im .* old.b)
        oriNew = angle.(sqrt.(norm.(eachrow([new.w new.b])).^2 .- (new.b .^2)) .+ im .* new.b)
    end

    # Create plot
    scatter(oriOld, ampOld, label = "Initial")
    scatter!(oriNew, ampNew, label = "Final")
    xlims!(-π, π)
    xlabel!("Orientation")
    ylabel!("Amplitude") |> display
end