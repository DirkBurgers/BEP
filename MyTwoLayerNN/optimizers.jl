# Interface name
abstract type AbstractOptimizer end

# --------------------------------------------------
# SGD Optimizer 
# --------------------------------------------------
struct SGDOptimzer{N<:Real} <: AbstractOptimizer
    η::N
end
SGDOptimzer(nn::TwoLayerNN, η) = SGDOptimzer(η)

function applyOptimizer!(o::SGDOptimzer, nn::TwoLayerNN, ∇data)
    @. nn.a -= o.η * ∇data.∇a
    @. nn.w -= o.η * ∇data.∇w
    @. nn.b -= o.η * ∇data.∇b
end

# --------------------------------------------------
# Adam Optimizer 
# --------------------------------------------------
struct AdamOptimizer{T<:Real, B<:Real, N<:Real} <: AbstractOptimizer
    ma::Vector{T}
    mw::Matrix{T}
    mb::Vector{T}
    va::Vector{T}
    vw::Matrix{T}
    vb::Vector{T}
    βₘ::B
    βᵥ::B
    βₚ::Vector{B}
    η::N
end
AdamOptimizer(nn::TwoLayerNN, η) = begin
    AdamOptimizer(
        zero(nn.a), zero(nn.w), zero(nn.b), 
        zero(nn.a), zero(nn.w), zero(nn.b),
        0.9, 0.999, [0.9, 0.999], η)
end

function applyOptimizer!(o::AdamOptimizer, nn::TwoLayerNN, ∇data)
    _adamstep!(o.ma, o.va, o.βₘ, o.βᵥ, o.βₚ, o.η, ∇data.∇a, nn.a)
    _adamstep!(o.mw, o.vw, o.βₘ, o.βᵥ, o.βₚ, o.η, ∇data.∇w, nn.w)
    _adamstep!(o.mb, o.vb, o.βₘ, o.βᵥ, o.βₚ, o.η, ∇data.∇b, nn.b)

    o.βₚ[1] *= o.βₘ
    o.βₚ[2] *= o.βᵥ
end

function _adamstep!(m, v, βₘ, βᵥ, βₚ, η, ∇, w) 
    @. m = βₘ * m + (1 - βₘ) * ∇
    @. v = βᵥ * v + (1 - βᵥ) * ∇ * ∇
    @fastmath @. w -=  η * m / (1 - βₚ[1]) / (√(v / (1 - βₚ[2])) + 1e-8)
end