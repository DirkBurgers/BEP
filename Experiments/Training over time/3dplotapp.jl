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
dataX = training_data.x
dataY = training_data.y

xmin, xmax = training_data.x |> Iterators.flatten |> extrema
xvals = range(xmin, xmax, length=100)

# ------------------------------------------------------
# ------------------ Helper functions ------------------
# ------------------------------------------------------
function deadneurons(nn)
    return [(b ≤ -w * xmin) && (b ≤ -w * xmax) ? :gray : MYORANGE for (w, b) in zip(nn.w, nn.b)]
end

function inflectionpoints(nn)
    return -nn.b ./ vec(nn.w)
end

function output(nn, vals)
    return [forward(nn, p) for p in vals]
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
deadobs = Observable(deadneurons(nn_data[1]))

# Line plot observables
yobs = Observable(output(nn_data[1], xvals))
iobs = Observable(inflectionpoints(nn_data[1]))
iyobs = Observable(output(nn_data[1], iobs[]))

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

    # Reset slider
    set_close_to!(sl, 1)
end

# Create time slider
sl = Slider(fig[3, :], range = 1:1:length(data["t_data"]), startvalue = 1)

lift(sl.value) do i
    # Update general observables
    tobs[] = t_data[i]

    # Update scatter plot observables
    wobs[] = vec(nn_data[i].w)
    bobs[] = nn_data[i].b
    aobs[] = nn_data[i].a
    deadobs[] = deadneurons(nn_data[i])

    # Update line plot observables
    yobs[] = output(nn_data[i], xvals)
    iobs[] = inflectionpoints(nn_data[i])
    iyobs[] = output(nn_data[i], iobs[])
end

# Create 3d Scatter plot
ax = Axis3(fig[2, 1], xlabel=L"w_k", ylabel=L"b_k", zlabel=L"a_k", viewmode=:fit)
scat = scatter!(ax, wobs, bobs, aobs, color=deadobs)

# Initial xyz limits
xyzmin = [-50.0, -50.0, -200.0]
xyzmax = [50.0, 50.0, 200.0]

xlims!(ax, xyzmin[1], xyzmax[1])
ylims!(ax, xyzmin[2], xyzmax[2])
zlims!(ax, xyzmin[3], xyzmax[3])

# Create initial line plot
ax_line = Axis(fig[2, 2])
lines!(ax_line, xvals, yobs, color=MYORANGE, linewidth=3)
scatter!(ax_line, vcat(dataX...), dataY, markersize = 16)
scatter!(ax_line, iobs, iyobs, marker=:star5, color=:darkgreen, markersize = 16)
xlims!(ax_line, xmin * 1.1, xmax * 1.1)

# Create figure title
Label(fig[0, :], @lift("m = $(length(nn_data[1].b)), α = $(nn_data[1].α), t = $(@sprintf "%.1e" $tobs)"), fontsize = 40)

# Create function to enable zooming with mouse wheel
on(events(ax.scene).scroll, priority=5) do (dx, dy)
    !is_mouseinside(ax.scene) && return

    zoomamount = [0.9, 0.9, 0.9]

    kb = events(ax.scene).keyboardbutton[]

    if kb.action == Keyboard.repeat
        if kb.key == Keyboard.x
            zoomamount = [0.9, 1.0, 1.0]
        elseif kb.key == Keyboard.y
            zoomamount = [1.0, 0.9, 1.0]
        elseif kb.key == Keyboard.z
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
    end
end

colsize!(fig.layout, 1, Relative(0.65))

# Start animation
display(fig)

