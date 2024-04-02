using Printf, LinearAlgebra
using JLD2, GLMakie

# Load MyTwoLayerNN if not loaded yet
(@isdefined MyTwoLayerNN) == false ?
begin
    include("../../MyTwoLayerNN/MyTwoLayerNN.jl")
    using .MyTwoLayerNN
    println("Included MyTwoLayerNN")
end : nothing

# ------------------------------------------------------
# --------------------- Settings -----------------------
# ------------------------------------------------------
DATA_FOLDER = "Training data"

# ------------------------------------------------------
# ------------------ Initialization --------------------
# ------------------------------------------------------
MYORANGE = Makie.RGB(213/255, 94/255, 0/255)

# Load data 
data = load(joinpath(@__DIR__, DATA_FOLDER, readdir(joinpath(@__DIR__, DATA_FOLDER))[1]))

t_data = data["t_data"]
nn_data = data["nn_data"]
training_data = data["training_data"]

# Extract data
xmin, xmax = training_data.x |> Iterators.flatten |> extrema
xvals = range(xmin, xmax, length=100)

# ------------------------------------------------------
# ------------------ Helper functions ------------------
# ------------------------------------------------------
function zone_neurons(nn)
    number_of_actived_datapoints = sum(x -> [b > -w * x for (w, b) in zip(nn.w, nn.b)], training_data.x |> Iterators.flatten |> collect)

    function tocolor(x)
        x == 0 && return :gray 
        x == 1 && return :purple
        x == 2 && return :blue
        x == 3 && return :green
        x == 4 && return MYORANGE
    end
    return number_of_actived_datapoints .|> tocolor
end

function inflectionpoints(nn)
    return -nn.b ./ vec(nn.w)
end

function output(nn, vals)
    return [forward(nn, p) for p in vals]
end

function inflection_color(nn)
    function tocolor(w, a)
        a <= 0 && w <= 0 && return :purple
        a <= 0 && return :red
        w <= 0 && return :blue
        w > 0 && return :darkgreen
    end
    
    return [tocolor(w, a) for (w, a) in zip(nn.w, nn.a)] 
end

# ------------------------------------------------------
# ------------------- Create 3D plot -------------------
# ------------------------------------------------------
fig = Figure()

# General observables
tobs = Observable(t_data[1])

# Scatter plot observables 
wobs = Observable(vec(nn_data[1].w))
bobs = Observable(nn_data[1].b)
aobs = Observable(nn_data[1].a)
zoneobs = Observable(zone_neurons(nn_data[1]))

# Line plot observables
yobs = Observable(output(nn_data[1], xvals))
iobs = Observable(inflectionpoints(nn_data[1]))
iyobs = Observable(output(nn_data[1], iobs[]))
icobs = Observable(inflection_color(nn_data[1]))

dataXobs = Observable(vcat(training_data.x...))
dataYobs = Observable(training_data.y)

# Options Menu
gopts = fig[1, 1:2] = GridLayout()
menu = Menu(gopts[1, 1], options = readdir(joinpath(@__DIR__, DATA_FOLDER)))

on(menu.selection) do selected_file
    # Load data
    data = load(joinpath(@__DIR__, DATA_FOLDER, selected_file))

    # Update data
    global t_data = data["t_data"]
    global nn_data = data["nn_data"]
    global training_data = data["training_data"]

    global xmin, xmax = training_data.x |> Iterators.flatten |> extrema
    global xvals = range(xmin, xmax, length=100)

    # Reset slider
    set_close_to!(sl, 1)
    sl.range = 1:1:length(t_data)

    # Update line plot
    updatelineplot()
end

# Create time slider
sl = Slider(fig[3, :], range = 1:1:length(t_data), startvalue = 1)

lift(sl.value) do i
    # Update general observables
    tobs[] = t_data[i]

    # Update scatter plot observables
    wobs[] = vec(nn_data[i].w)
    bobs[] = nn_data[i].b
    aobs[] = nn_data[i].a
    zoneobs[] = zone_neurons(nn_data[i])

    # Update line plot observables
    yobs[] = output(nn_data[i], xvals)
    iobs[] = inflectionpoints(nn_data[i])
    iyobs[] = output(nn_data[i], iobs[])
    icobs[] = inflection_color(nn_data[i])
end

# Create 3d Scatter plot
ax = Axis3(fig[2, 1], xlabel=L"w_k", ylabel=L"b_k", zlabel=L"a_k", viewmode=:fit)
scat = scatter!(ax, wobs, bobs, aobs, color=zoneobs)

# Initial xyz limits
xyzmin = [-50.0, -50.0, -200.0]
xyzmax = [50.0, 50.0, 200.0]

xlims!(ax, xyzmin[1], xyzmax[1])
ylims!(ax, xyzmin[2], xyzmax[2])
zlims!(ax, xyzmin[3], xyzmax[3])

# Create initial line plot
ax_line = Axis(fig[2, 2])
lines!(ax_line, xvals, yobs, color=MYORANGE, linewidth=3)
scat_data = scatter!(ax_line, dataXobs, dataYobs, markersize = 16)
scatter!(ax_line, iobs, iyobs, marker=:star5, color=icobs, markersize = 16)
xlims!(ax_line, xmin * 1.1, xmax * 1.1)

function updatelineplot()
    # Update training data points
    dataXobs[] = vcat(training_data.x...)
    dataYobs[] = training_data.y

    # Update axis 
    autolimits!(ax_line)
    xlims!(ax_line, xmin * 1.1, xmax * 1.1)
end

# Create figure title
Label(fig[0, :], @lift("m = $(length(nn_data[1].b)), α = $(nn_data[1].α), t = $(@sprintf "%.1e" $tobs), η = $(training_data.learning_rate)"), fontsize = 40)

# Create function to enable zooming with mouse wheel
on(events(ax.scene).scroll, priority=5) do (dx, dy)
    !is_mouseinside(ax.scene) && return

    zoomamount = [0.9, 0.9, 0.9]

    kb = events(ax.scene).keyboardbutton[]

    if kb.action == Keyboard.repeat
        if kb.key == Keyboard.x || kb.key == Keyboard.w
            zoomamount = [0.9, 1.0, 1.0]
        elseif kb.key == Keyboard.y || kb.key == Keyboard.b
            zoomamount = [1.0, 0.9, 1.0]
        elseif kb.key == Keyboard.z || kb.key == Keyboard.a
            zoomamount = [1.0, 1.0, 0.9]
        end
    end 

    if dy > 0
        xyzmin .*= zoomamount
        xyzmax .*= zoomamount    
    elseif dy < 0
        xyzmin ./= zoomamount
        xyzmax ./= zoomamount
    end

    if !iszero(dy)
        xlims!(ax, xyzmin[1], xyzmax[1])
        ylims!(ax, xyzmin[2], xyzmax[2])
        zlims!(ax, xyzmin[3], xyzmax[3])
    end
end

# Go forward/backward in time with arrow keys
on(events(ax.scene).keyboardbutton) do event
    !(event.action == Keyboard.press || event.action == Keyboard.repeat) && return

    if event.key == Keyboard.right
        set_close_to!(sl, sl.value[] + 1)
    elseif event.key == Keyboard.left
        set_close_to!(sl, sl.value[] - 1)
    elseif event.key == Keyboard.u 
        ax.azimuth = -0.5π
        ax.elevation = π/2
    elseif event.key == Keyboard.l
        ax.azimuth = π
        ax.elevation = 0.0
    elseif event.key == Keyboard.f
        ax.azimuth = π/2
        ax.elevation = 0.0
    end
end

colsize!(fig.layout, 1, Relative(0.65))

# Start animation
display(fig)

