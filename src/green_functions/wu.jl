using SpecialFunctions: besselj0, besselj1, bessely0, bessely1

struct GFWu <: GreensFunction end

# Scalar wave term (Liang–Wu–Noblesse). Named functions so broadcast / AD
# see a stable type instead of a per-pair `do` closure.
@inline function _wu_reduced_hh_vv(x, xi, wavenumber)
    dis_squared_raw = (x[1] - xi[1])^2 + (x[2] - xi[2])^2
    purt = oftype(dis_squared_raw, 1e-18)
    if dis_squared_raw == 0
        dis = sqrt(dis_squared_raw + purt^2) - purt
    else
        dis = sqrt(dis_squared_raw)
    end
    hh = wavenumber * dis
    vv = wavenumber * (x[3] + xi[3])
    return hh, vv
end

@inline function _wu_wave_term(hh, vv)
    dd = sqrt(hh * hh + vv * vv)
    alpha = -vv / dd
    beta = hh / dd
    rho = dd / (1 + dd)
    return -GF_Func_L0(vv, dd, alpha, beta, rho) - GF_Func_W(hh, vv)
end

@inline function _wu_wave_term_and_derivs(hh, vv)
    dd = sqrt(hh * hh + vv * vv)
    alpha = -vv / dd
    beta = hh / dd
    rho = dd / (1 + dd)
    GF = -GF_Func_L0(vv, dd, alpha, beta, rho) - GF_Func_W(hh, vv)
    dGF_dhh = -GF_Func_Ls(hh, vv, dd, alpha, beta, rho) - GF_Func_Wh(hh, vv)
    dGF_dvv = GF + 2 / dd
    return GF, dGF_dhh, dGF_dvv
end

@inline function _wu_greens(x, xi, wavenumber)
    hh, vv = _wu_reduced_hh_vv(x, xi, wavenumber)
    return wavenumber * _wu_wave_term(hh, vv)
end

function greens(::GFWu, element_1, element_2, wavenumber)
    _wu_greens(center(element_1), center(element_2), wavenumber)
end

function gradient_greens(gf::GFWu, element_1, element_2, wavenumber; with_respect_to_first_variable=false)
    both_greens_and_gradient_greens(gf, element_1, element_2, wavenumber; with_respect_to_first_variable)[2]
end

function both_greens_and_gradient_greens(g::GFWu, element_1, element_2, wavenumber; with_respect_to_first_variable=false)
    x = center(element_1)
    xi = center(element_2)
    hh, vv = _wu_reduced_hh_vv(x, xi, wavenumber)
    GF, dGF_dhh, dGF_dvv = _wu_wave_term_and_derivs(hh, vv)
    k = wavenumber
    if with_respect_to_first_variable
        if abs(hh) > 1e-6
            dr_dx1 = k^2 / hh * (x[1] - xi[1])
            dr_dx2 = k^2 / hh * (x[2] - xi[2])
        else
            dr_dx1 = zero(x[1])
            dr_dx2 = zero(x[2])
        end
        return k * GF, k * (zero(x) .+ (dr_dx1 * dGF_dhh, dr_dx2 * dGF_dhh, k * dGF_dvv))
    else
        if abs(hh) > 1e-6
            dr_dxi1 = k^2 / hh * (xi[1] - x[1])
            dr_dxi2 = k^2 / hh * (xi[2] - x[2])
        else
            dr_dxi1 = zero(x[1])
            dr_dxi2 = zero(x[2])
        end
        return k * GF, k * (zero(x) .+ (dr_dxi1 * dGF_dhh, dr_dxi2 * dGF_dhh, k * dGF_dvv))
    end
end

@inline function integral(::GFWu, element_1, element_2, wavenumber)
    _wu_greens(center(element_1), center(element_2), wavenumber) * area(element_2)
end

@inline function integral_gradient(g::GFWu, element_1, element_2, wavenumber; with_respect_to_first_variable=false)
    gradient_greens(g, element_1, element_2, wavenumber; with_respect_to_first_variable) * area(element_2)
end

function both_integral_and_integral_gradient(g::GFWu, element_1, element_2, wavenumber; with_respect_to_first_variable=false)
    GF, dGF = both_greens_and_gradient_greens(g, element_1, element_2, wavenumber; with_respect_to_first_variable)
    a = area(element_2)
    return GF * a, dGF * a
end

@inline _wu_integral_centers(c1, c2, area2, k) = _wu_greens(c1, c2, k) * area2

# One Bessel/Struve evaluation per pair; S and D share `_wu_wave_term_and_derivs`.
# `direct` is a compile-time Val so the kernel does not carry a Bool branch.
@inline _wu_ndot_gradient_centers(c1, c2, n, area2, k, direct::Bool) =
    _wu_ndot_gradient_centers(c1, c2, n, area2, k, Val(direct))

@inline function _wu_ndot_gradient_centers(c1, c2, n, area2, k, ::Val{direct}) where {direct}
    _wu_sd_centers(c1, c2, n, area2, k, Val(direct))[2]
end

@inline function _wu_sd_centers(c1, c2, n, area2, k, ::Val{true})
    _wu_sd(c1, c2, n, area2, k, c2[1] - c1[1], c2[2] - c1[2])
end

@inline function _wu_sd_centers(c1, c2, n, area2, k, ::Val{false})
    _wu_sd(c1, c2, n, area2, k, c1[1] - c2[1], c1[2] - c2[2])
end

@inline function _wu_sd(x, xi, n, area2, k, dx1, dx2)
    hh, vv = _wu_reduced_hh_vv(x, xi, k)
    GF, dGF_dhh, dGF_dvv = _wu_wave_term_and_derivs(hh, vv)
    if abs(hh) > 1e-6
        dr1 = k^2 / hh * dx1
        dr2 = k^2 / hh * dx2
    else
        z = zero(hh)
        dr1 = z
        dr2 = z
    end
    g = k * (zero(x) .+ (dr1 * dGF_dhh, dr2 * dGF_dhh, k * dGF_dvv))
    s = (k * GF) * area2
    d = (n' * g) * area2
    return s, d
end

######################################################

@inline function GF_dFuncC(tt)
     C0 = 14.19
     C1 = -148.24
     C2 = 847.8
     C3 = -2318.58
     C4 = 3168.35
    C5 = -1590.27

    GF_dFuncC = (C4 + C5 * tt) * tt
    GF_dFuncC = (C3 + GF_dFuncC) * tt
    GF_dFuncC = (C2 + GF_dFuncC) * tt
    GF_dFuncC = (C1 + GF_dFuncC) * tt
    GF_dFuncC = C0 + GF_dFuncC

    return GF_dFuncC
end


@inline function GF_dFuncB(tt)
    B0 = 1.11
    B1 = 2.894
    B2 = -76.765
    B3 = 1565.35
    B4 = -11336.19
    B5 = 44270.15
    B6 = -97014.11
    B7 = 118879.26
    B8 = -76209.82
    B9 = 19923.28

    GF_dFuncB = (B8 + B9 * tt) * tt
    GF_dFuncB = (B7 + GF_dFuncB) * tt
    GF_dFuncB = (B6 + GF_dFuncB) * tt
    GF_dFuncB = (B5 + GF_dFuncB) * tt
    GF_dFuncB = (B4 + GF_dFuncB) * tt
    GF_dFuncB = (B3 + GF_dFuncB) * tt
    GF_dFuncB = (B2 + GF_dFuncB) * tt
    GF_dFuncB = (B1 + GF_dFuncB) * tt
    GF_dFuncB = B0 + GF_dFuncB

    return GF_dFuncB
end

@inline function GF_dFuncA(tt)
    A0 = 2.948
    A1 = -24.53
    A2 = 249.69
    A3 = -754.85
    A4 = -1187.71
    A5 = 16370.75
    A6 = -48811.41
    A7 = 68220.87
    A8 = -46688.0
    A9 = 12622.25

    GF_dFuncA = (A8 + A9 * tt) * tt
    GF_dFuncA = (A7 + GF_dFuncA) * tt
    GF_dFuncA = (A6 + GF_dFuncA) * tt
    GF_dFuncA = (A5 + GF_dFuncA) * tt
    GF_dFuncA = (A4 + GF_dFuncA) * tt
    GF_dFuncA = (A3 + GF_dFuncA) * tt
    GF_dFuncA = (A2 + GF_dFuncA) * tt
    GF_dFuncA = (A1 + GF_dFuncA) * tt
    GF_dFuncA = A0 + GF_dFuncA

    return GF_dFuncA
end

#import Pkg; Pkg.add("SpecialFunctions")

@inline function GF_Func_Wh(hh, vv)
    H1 = StruveH1(hh)
    J1 = besselj1(hh)
    return 2 * pi * (2 / pi - H1 + im * J1) * exp(vv)
end

@inline function GF_Func_Lsp(hh, vv, dd, alpha, beta, rho)
    A = GF_dFuncA(rho)
    B = GF_dFuncB(rho)
    C = GF_dFuncC(rho)

    RR = beta * A
    RR -= (1.0 - alpha) * B
    RR += beta * (1.0 - beta) * rho * (1.0 - 2.0 * rho) * C

    GF_Func_Lsp = rho * (1.0 - rho)^3 * RR

    return GF_Func_Lsp
end

@inline function GF_Func_Ls(hh, vv, dd, alpha, beta, rho)
    PS = (beta + hh) / (dd - vv)
    PS = PS - 2.0 * beta + 2.0 * exp(vv) * dd - hh

    QS = exp(-dd) * (1.0 - beta)
    QS = QS * (1.0 + dd / (1.0 + dd^3))

    Lsp = GF_Func_Lsp(hh, vv, dd, alpha, beta, rho)

    GF_Func_Ls = 2.0 * PS / (1.0 + dd^3) - 4.0 * QS + 2.0 * Lsp

    return GF_Func_Ls
end

@inline function GF_FuncD(tt)
    D0 = 0.632
    D1 = -40.97
    D2 = 667.16
    D3 = -6072.07
    D4 = 31127.39
    D5 = -96293.05
    D6 = 181856.75
    D7 = -205690.43
    D8 = 128170.2
    D9 = -33744.6

    GF_FuncD = (D8 + D9 * tt) * tt
    GF_FuncD = (D7 + GF_FuncD) * tt
    GF_FuncD = (D6 + GF_FuncD) * tt
    GF_FuncD = (D5 + GF_FuncD) * tt
    GF_FuncD = (D4 + GF_FuncD) * tt
    GF_FuncD = (D3 + GF_FuncD) * tt
    GF_FuncD = (D2 + GF_FuncD) * tt
    GF_FuncD = (D1 + GF_FuncD) * tt
    GF_FuncD = D0 + GF_FuncD

    return GF_FuncD
end

@inline function GF_FuncC(tt)
    C0 = 1.268
    C1 = -9.747
    C2 = 209.653
    C3 = -1397.89
    C4 = 5155.67
    C5 = -9844.35
    C6 = 9136.4
    C7 = -3272.62

    GF_FuncC = (C6 + C7 * tt) * tt
    GF_FuncC = (C5 + GF_FuncC) * tt
    GF_FuncC = (C4 + GF_FuncC) * tt
    GF_FuncC = (C3 + GF_FuncC) * tt
    GF_FuncC = (C2 + GF_FuncC) * tt
    GF_FuncC = (C1 + GF_FuncC) * tt
    GF_FuncC = C0 + GF_FuncC

    return GF_FuncC
end

@inline function GF_FuncB(tt)
    B0 = 0.938
    B1 = 5.737
    B2 = -67.92
    B3 = 796.534
    B4 = -4780.77
    B5 = 17137.74
    B6 = -36618.81
    B7 = 44894.06
    B8 = -29030.24
    B9 = 7671.22

    GF_FuncB = (B8 + B9 * tt) * tt
    GF_FuncB = (B7 + GF_FuncB) * tt
    GF_FuncB = (B6 + GF_FuncB) * tt
    GF_FuncB = (B5 + GF_FuncB) * tt
    GF_FuncB = (B4 + GF_FuncB) * tt
    GF_FuncB = (B3 + GF_FuncB) * tt
    GF_FuncB = (B2 + GF_FuncB) * tt
    GF_FuncB = (B1 + GF_FuncB) * tt
    GF_FuncB = B0 + GF_FuncB

    return GF_FuncB
end

@inline function GF_FuncA(tt)
    A0 = 1.21
    A1 = -13.328
    A2 = 215.896
    A3 = -1763.96
    A4 = 8418.94
    A5 = -24314.21
    A6 = 42002.57
    A7 = -41592.9
    A8 = 21859.0
    A9 = -4838.6

    GF_FuncA = (A8 + A9 * tt) * tt
    GF_FuncA = (A7 + GF_FuncA) * tt
    GF_FuncA = (A6 + GF_FuncA) * tt
    GF_FuncA = (A5 + GF_FuncA) * tt
    GF_FuncA = (A4 + GF_FuncA) * tt
    GF_FuncA = (A3 + GF_FuncA) * tt
    GF_FuncA = (A2 + GF_FuncA) * tt
    GF_FuncA = (A1 + GF_FuncA) * tt
    GF_FuncA = A0 + GF_FuncA

    return GF_FuncA
end



@inline function GF_Func_W(hh, vv)
    H0 = StruveH0(hh)
    J0 = besselj0(hh)
    return 2 * pi * complex(H0, -J0) * exp(vv)
end

@inline function GF_Func_Lp(alpha, beta, rho)
    A = GF_FuncA(rho)
    B = GF_FuncB(rho)
    C = GF_FuncC(rho)
    D = GF_FuncD(rho)

    RR = (1.0 - beta) * A
    RR -= beta * B
    RR -= alpha * C / (1.0 + 6.0 * alpha * rho * (1.0 - rho))
    RR += beta * (1.0 - beta) * D

    GF_Func_Lp = rho * (1.0 - rho)^3 * RR

    return  GF_Func_Lp
end

@inline function GF_Func_L0(vv, dd, alpha, beta, rho)
    gama = 0.5772156649
    PP = log(0.5 * (dd - vv)) + gama - 2.0 * dd^2
    PP = exp(vv) * PP
    PP = PP + dd^2 - vv

    Lp = GF_Func_Lp( alpha, beta, rho)

    GF_Func_L0 = 2.0 * PP / (1.0 + dd^3) + 2.0 * Lp

    return   GF_Func_L0
end

@inline function StruveH0(xx)
    if xx <= 3.0
        yy = (xx / 3.0)^2
        
        P0 = +1.909859164
        P1 = -1.909855001
        P2 = +0.687514637
        P3 = -0.126164557
        P4 = +0.013828813
        P5 = -0.000876918
        
        StruveH0 = P0 + (P1 + (P2 + (P3 + (P4 + P5 * yy) * yy) * yy) * yy) * yy
        StruveH0 *= xx / 3.0
    else
        yy = (3.0 / xx)^2
        
        a0 = 0.99999906
        a1 = 4.77228920
        a2 = 3.85542044
        a3 = 0.32303607

        b1 = 4.88331068
        b2 = 4.28957333
        b3 = 0.52120508

        c1 = 2.0 * (a0 + (a1 + (a2 + a3 * yy) * yy) * yy)
        c2 = pi * xx * (1.0 + (b1 + (b2 + b3 * yy) * yy) * yy)
        
        StruveH0 = c1 / c2 + bessely0(xx)
    end
    
    return StruveH0
end



@inline function StruveH1(xx)
    if xx <= 3.0
        yy = (xx / 3.0)^2
        
        P1 = +1.909859286
        P2 = -1.145914713
        P3 = +0.294656958
        P4 = -0.042070508
        P5 = +0.003785727
        P6 = -0.000207183
      
        StruveH1 = (P1 + (P2 + (P3 + (P4 + (P5 + P6 * yy) * yy) * yy) * yy) * yy) * yy
    else
        yy = (3.0 / xx)^2

        a0 = 1.00000004
        a1 = 3.92205313
        a2 = 2.64893033
        a3 = 0.27450895

        b1 = 3.81095112
        b2 = 2.26216956
        b3 = 0.10885141

        c1 = 2.0 * (a0 + (a1 + (a2 + a3 * yy) * yy) * yy)
        c2 = pi * (1.0 + (b1 + (b2 + b3 * yy) * yy) * yy)

        StruveH1 = c1 / c2 + bessely1(xx)
    end
    
    return StruveH1
end
