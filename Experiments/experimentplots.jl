using LinearAlgebra
using Makie, CairoMakie

set_theme!(theme_latexfonts()) # Theme for plots 
update_theme!(Theme(fontsize = 18)) # 36

function createlineplot!(ax::Axis, nn::MyTwoLayerNN.TwoLayerNN, td::MyTwoLayerNN.TrainingData)
    xmin, xmax = td.x |> Iterators.flatten |> extrema
    
    # Calculate points on the line 
    x = range(xmin, xmax, length=100)
    y = [MyTwoLayerNN.forward(nn, p) for p in x]

    lines!(ax, x, y, color=:darkorange2)
    
    # Plot the points
    scatter!(ax, vcat(td.x...), td.y)
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
    scat_init = scatter!(ax, ori_initial, amp_initial)
    scat_trained = scatter!(ax, ori_trained, amp_trained, color=:darkorange2)
    
    axislegend(ax, [scat_init, scat_trained], ["Before training", "After training"])
end

function createorientiationplot(nni, nnt)
    # Initialize figure 
    f = Figure()
    ax = Axis(f[1, 1], xticks=-3:1:3, xlabel="Orientation", ylabel="Amplitude")
    
    xlims!(ax, -π, π)
    
    # Calculate amplitude
    amp_initial = abs.(nni.a) .* norm.(eachrow([nni.w nni.b]))
    amp_trained = abs.(nnt.a) .* norm.(eachrow([nnt.w nnt.b]))
    
    # Calculate orientation
    ori_initial = angle.(vec(nni.w) + im * nni.b)
    ori_trained = angle.(vec(nnt.w) + im * nnt.b)
    
    # Plot all points
    scat_init = scatter!(ax, ori_initial, amp_initial, strokewidth=0.5)
    scat_trained = scatter!(ax, ori_trained, amp_trained, color=:darkorange2, strokewidth=0.5)
    
    axislegend(ax, [scat_init, scat_trained], ["Before training", "After training"])
    
    return f
end