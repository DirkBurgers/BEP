using LinearAlgebra
using Makie, CairoMakie

set_theme!(theme_latexfonts()) # Theme for plots 
update_theme!(Theme(fontsize = 36)) # 18 for testing and 36 for report

function createlineplot!(ax::Axis, nn::MyTwoLayerNN.TwoLayerNN, td::MyTwoLayerNN.TrainingData)
    xmin, xmax = td.x |> Iterators.flatten |> extrema
    
    # Calculate points on the line 
    x = range(xmin, xmax, length=100)
    y = [MyTwoLayerNN.forward(nn, p) for p in x]

    lines!(ax, x, y, color=:darkorange2, linewidth=3)
    
    # Plot the points
    scatter!(ax, vcat(td.x...), td.y, markersize=16)
end

function createorientiationplot!(ax::Axis, nni, nnt)
    # Set axis labels
    ax.xlabel = "Orientation"
    ax.ylabel = "Amplitude"
    
    xlims!(ax, -π, π)
    
    # Calculate amplitude
    amp_initial = abs.(nni.a) .* norm.(eachrow([nni.w nni.b]))
    amp_trained = abs.(nnt.a) .* norm.(eachrow([nnt.w nnt.b]))
    
    # Calculate orientation
    ori_initial = angle.(vec(nni.w) + im * nni.b)
    ori_trained = angle.(vec(nnt.w) + im * nnt.b)
    
    # Plot all points
    scat_init = scatter!(ax, ori_initial, amp_initial, markersize=16)
    scat_trained = scatter!(ax, ori_trained, amp_trained, markersize=16, color=:darkorange2)
    
    axislegend(ax, [scat_init, scat_trained], ["Before training", "After training"])
end

function createorientiationplot(nni, nnt)
    f = Figure()
    ax = Axis(f[1, 1])
    createorientiationplot!(ax, nni, nnt)
    return f
end