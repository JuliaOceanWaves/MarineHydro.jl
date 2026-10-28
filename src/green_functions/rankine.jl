"""
    DelhommeauRankine()

Previous Rankine panel integrals (Capytaine / Delhommeau near-field + far-field `A/r`).

`Rankine()` is now the Birk vectorized kernel. Keep this type for Capytaine
comparisons and for checking the two Rankine implementations against each other.
"""
struct DelhommeauRankine <: GreensFunction end
wavenumber_independent(::DelhommeauRankine) = true

_distance(x, ξ) = norm(x .- ξ)

# All functions below depends on wavenumber but the variable is actually unused.
# It is just part of the signature to have the same signature than the wave part of the Green function.

function greens(::DelhommeauRankine, element_1, element_2, wavenumber=nothing)
    r = _distance(center(element_1), center(element_2))
    return 1 / r
end

function gradient_greens(::DelhommeauRankine, element_1, element_2, wavenumber=nothing; with_respect_to_first_variable=false)
    r = _distance(center(element_1), center(element_2))
    r̅ = center(element_2) - center(element_1)
    ∇ₓ = 1/r^3*r̅
    if with_respect_to_first_variable
        return ∇ₓ
    else  # Gradient with respect to second variable
        return -∇ₓ
    end
end

function integral(::DelhommeauRankine, element_1, element_2, wavenumber=nothing)
    point = center(element_1)
    source_point = center(element_2)
    source_vertices = vertices(element_2)
    source_radius = radius(element_2)
    source_normal = normal(element_2)
    source_area = area(element_2)
    r̄ = point - source_point
    r = norm(r̄)
    if r > 7 * source_radius # if far -> Rankine Direct
        integral = source_area / r
    else  # else (if close) deal with singularity
        integral = zero(source_area)
        GZ = dot(r̄, source_normal)
        RR = ntuple(i -> _distance(point, source_vertices[i,:]), 4)
        for index_vertex in 1:4
            current_vertex = source_vertices[index_vertex, :]
            index_next_vertex = mod1(index_vertex+1, 4)
            next_vertex = source_vertices[index_next_vertex, :]
            current_to_next = next_vertex - current_vertex
            segment_length = norm(current_to_next)
            if segment_length >= 1e-3 * source_radius
                GY = dot(point - current_vertex, cross(source_normal, current_to_next / segment_length))
                ANT = 2 * GY * segment_length
                DNT = (RR[index_next_vertex] + RR[index_vertex])^2 - segment_length^2 +
                    2 * _abs_ad(GZ) * (RR[index_next_vertex] + RR[index_vertex])
                AT = _delhommeau_atan(ANT, DNT, GZ, source_radius)
                integral -= 2 * _abs_ad(GZ) * AT
                if abs(GY) > 1e-5
                    ANL = RR[index_next_vertex] + RR[index_vertex] + segment_length
                    DNL = RR[index_next_vertex] + RR[index_vertex] - segment_length
                    ALDEN = log(ANL / DNL)
                    integral += GY * ALDEN
                end
            end
        end
    end
    return integral
end

function both_integral_and_integral_gradient(::DelhommeauRankine, element_1, element_2, wavenumber=nothing; with_respect_to_first_variable=false)
    point = center(element_1)
    source_point = center(element_2)
    source_vertices = vertices(element_2)
    source_normal = normal(element_2)
    source_radius = radius(element_2)
    source_area = area(element_2)
    r̄ = point - source_point
    r = norm(r̄)
    if r > 7 * source_radius # if far -> Rankine Direct
        derivative = 1 / r^3 * r̄
        integral = source_area / r
        integral_gradient = derivative * source_area
    else  # else (if close) deal with singularity
        integral = zero(source_area)
        integral_gradient = zero(point)
        GZ = dot(r̄, source_normal)
        RR = ntuple(i -> _distance(point, source_vertices[i,:]), 4)
        DRX = ntuple(i -> (point - source_vertices[i,:]) / RR[i], 4)
        for index_vertex in 1:4
            current_vertex = source_vertices[index_vertex,:]
            index_next_vertex = mod1(index_vertex+1, 4)
            next_vertex = source_vertices[index_next_vertex, :]
            current_to_next = next_vertex - current_vertex
            segment_length = norm(current_to_next)
            if segment_length >= 1e-3 * source_radius
                GYX = cross(source_normal, current_to_next / segment_length)
                GY = dot(point - current_vertex, GYX)
                ANT = 2 * GY * segment_length
                DNT = (RR[index_next_vertex] + RR[index_vertex])^2 - segment_length^2 +
                    2 * _abs_ad(GZ) * (RR[index_next_vertex] + RR[index_vertex])
                ANL = RR[index_next_vertex] + RR[index_vertex] + segment_length
                DNL = RR[index_next_vertex] + RR[index_vertex] - segment_length
                ALDEN = log(ANL / DNL)
                AT = _delhommeau_atan(ANT, DNT, GZ, source_radius)
                #error bound error in the line below...
                ANLX = DRX[index_next_vertex] + DRX[index_vertex]
                ANTX = 2 * segment_length * GYX
                DNTX = 2 * (RR[index_next_vertex] + RR[index_vertex] + _abs_ad(GZ)) * ANLX +
                    2 * _sign_ad(GZ) * (RR[index_next_vertex] + RR[index_vertex]) * source_normal

                # Same S as `integral(::DelhommeauRankine)`: solid-angle atan plus log term.
                integral -= 2 * _abs_ad(GZ) * AT
                if abs(GY) > 1e-5
                    integral += GY * ALDEN
                end
                integral_gradient = integral_gradient .-
                    (ALDEN .* GYX .- 2 * _sign_ad(GZ) * AT .* source_normal .+
                    GY * (DNL - ANL) ./ (ANL .* DNL) .* ANLX .-
                    2 * _abs_ad(GZ) .* (ANTX .* DNT .- DNTX .* ANT) ./ (ANT .* ANT .+ DNT .* DNT))
            end
        end
    end
    if with_respect_to_first_variable
        return integral, -integral_gradient
    else  # Gradient with respect to second variable
        return integral, integral_gradient
    end
end


function integral_gradient(gf::DelhommeauRankine, element_1, element_2, wavenumber=nothing; with_respect_to_first_variable=false)
    both_integral_and_integral_gradient(gf, element_1, element_2, wavenumber; with_respect_to_first_variable)[2]
end
