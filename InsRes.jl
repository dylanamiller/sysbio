module InsRes

using DifferentialEquations
using Plots
using DataFrames

export example, parameter_sensitivity, plot_results, healthy_initial_conditions, insulin_resistant_initial_conditions, default_parameters, run_simulation

"""
Alzheimer's Disease Energy Metabolism Model
Based on the interactions between insulin signaling, amyloid-β, tau phosphorylation,
and glucose metabolism including O-GlcNAcylation pathway.
"""

# Define the ODE system
function ad_metabolism_model!(du, u, p, t)
    # Extract state variables
    I, Aβ, τp, Inf, AGE, OGlcNAc, GLUT1 = u
    
    # Extract parameters (destructured for clarity)
    # Insulin parameters
    αI = p[1]        # Basal insulin production rate (nM/hr)
    βI = p[2]        # IDE-mediated insulin degradation rate constant (1/hr)
    KI = p[3]        # Michaelis constant for insulin-IDE binding (nM)
    γI = p[4]        # Insulin-Aβ binding rate (1/(nM·hr))
    δI = p[5]        # BBB transport rate constant (1/hr)
    KBBB = p[6]      # BBB transport saturation constant (nM)
    
    # Amyloid-β parameters
    αAβ = p[7]       # Basal Aβ production rate (nM/hr)
    βAβ = p[8]       # IDE-mediated Aβ degradation rate constant (1/hr)
    KAβ = p[9]       # Michaelis constant for Aβ-IDE binding (nM)
    κ = p[10]        # Competitive inhibition strength of insulin on Aβ-IDE binding
    δAβ = p[11]      # Non-IDE Aβ clearance rate (1/hr)
    μ = p[12]        # Inflammation effect on Aβ (nM⁻¹)
    ξ = p[13]        # Trans-Golgi contribution factor
    KTG = p[14]      # Trans-Golgi saturation constant
    TG = p[15]       # Trans-Golgi network activity (normalized, 0-1)
    
    # Tau phosphorylation parameters
    kf = p[16]       # Tau phosphorylation rate constant (1/hr)
    kr = p[17]       # Tau dephosphorylation rate constant (1/hr)
    τ0 = p[18]       # Total tau protein concentration (nM)
    φ = p[19]        # GSK-3β activity coefficient
    ψOG = p[20]      # O-GlcNAc effect on tau phosphorylation
    
    # Inflammation parameters
    ρ = p[21]        # Aβ-induced inflammation rate (hr⁻¹)
    σ = p[22]        # AGE-induced inflammation rate (nM⁻¹·hr⁻¹)
    λInf = p[23]     # Inflammation resolution rate (hr⁻¹)
    KInf = p[24]     # Half-saturation constant for Aβ-induced inflammation (nM)
    
    # AGE parameters
    νAGE = p[25]     # AGE formation rate from hyperglycemia (hr⁻¹)
    δAGE = p[26]     # AGE clearance rate (hr⁻¹)
    HG = p[27]       # Hyperglycemia factor (normalized, 1 = normal)
    
    # O-GlcNAcylation parameters
    kHBP = p[28]     # HBP pathway rate constant (hr⁻¹)
    G = p[29]        # Intracellular glucose concentration (mM)
    GFAT = p[30]     # GFAT enzyme level (normalized)
    KGFAT = p[31]    # Michaelis constant for GFAT
    kOGA = p[32]     # OGA enzyme activity (hr⁻¹)
    OGA = p[33]      # OGA enzyme level (normalized)
    KOG = p[34]      # Michaelis constant for OGA (nM)
    
    # GLUT1 parameters
    αGLUT1 = p[35]   # GLUT1 production rate (normalized units/hr)
    δGLUT1 = p[36]   # GLUT1 degradation rate (hr⁻¹)
    θTX = p[37]      # TXNIP inhibition strength on GLUT1
    TXNIP0 = p[38]   # Basal TXNIP level (normalized)
    νTX = p[39]      # Akt inhibition strength on TXNIP
    
    # Insulin signaling parameters
    KIS = p[40]      # Half-saturation constant for insulin signaling (nM)
    η = p[41]        # Aβ effect on insulin signaling
    ψInf = p[42]     # Inflammation effect on insulin signaling
    χ = p[43]        # PTEN effect on insulin signaling
    κAkt = p[44]     # Akt activation coefficient
    
    # Additional signaling parameters
    GSK3β0 = p[45]   # Basal GSK-3β activity (normalized)
    θGSK = p[46]     # Insulin inhibition of GSK-3β
    PP2A = p[47]     # PP2A phosphatase level (normalized)
    PP1 = p[48]      # PP1 phosphatase level (normalized)
    Kτ = p[49]       # Half-saturation for phosphatase activity (nM)
    
    # PTEN-tau feedback parameters
    PTEN0 = p[50]    # Basal PTEN activity (normalized)
    ωτ = p[51]       # Tau inhibition of PTEN
    Kτ0 = p[52]      # Half-saturation for tau effect on PTEN (nM)
    
    # IDE level
    IDE = p[53]      # Insulin-degrading enzyme concentration (normalized)
    
    # Calculate helper variables
    
    # PTEN activity (inhibited by tau)
    PTEN_active = PTEN0 * (1 - ωτ * τ0 / (τ0 + Kτ0))
    
    # Insulin signaling cascade
    IS = (I / (I + KIS)) * 
         (1 / (1 + η * Aβ)) * 
         (1 / (1 + ψInf * Inf)) * 
         (1 / (1 + χ * (1 - PTEN_active)))
    
    # Akt activation
    Akt_active = κAkt * IS
    
    # TXNIP regulation by Akt
    TXNIP = TXNIP0 / (1 + νTX * Akt_active)
    
    # GSK-3β activity (inhibited by insulin signaling)
    GSK3β_active = GSK3β0 / (1 + θGSK * IS)
    
    # Phosphatase activity (enhanced by insulin signaling)
    PP_active = (PP2A + PP1) * (IS / (IS + Kτ))
    
    # Differential equations
    
    # 1. Insulin dynamics
    du[1] = αI - 
            βI * IDE * I / (KI + I) - 
            γI * I * Aβ - 
            δI * I / (KBBB + I)
    
    # 2. Amyloid-β dynamics
    du[2] = αAβ * (1 + ξ * TG / KTG) - 
            βAβ * IDE * Aβ / (KAβ + Aβ + κ * I) - 
            δAβ * Aβ - 
            μ * Inf * Aβ
    
    # 3. Phosphorylated tau dynamics
    du[3] = kf * τ0 * (1 + φ * GSK3β_active) * (1 + ψOG / (OGlcNAc + 0.1)) - 
            kr * τp * PP_active
    
    # 4. Inflammation dynamics
    du[4] = ρ * (Aβ / (KInf + Aβ)) + 
            σ * AGE - 
            λInf * Inf
    
    # 5. AGE dynamics
    du[5] = νAGE * HG - δAGE * AGE
    
    # 6. O-GlcNAcylation dynamics
    du[6] = kHBP * G * (GFAT / (KGFAT + GFAT)) - 
            kOGA * OGA * OGlcNAc / (KOG + OGlcNAc)
    
    # 7. GLUT1 dynamics
    du[7] = αGLUT1 * (1 / (1 + TXNIP * θTX)) - δGLUT1 * GLUT1
end

# Define default parameters
function default_parameters()
    p = zeros(53)
    
    # Insulin parameters
    p[1] = 10.0      # αI: Basal insulin production rate (nM/hr)
    p[2] = 2.0       # βI: IDE-mediated insulin degradation rate (1/hr)
    p[3] = 50.0      # KI: Michaelis constant for insulin-IDE (nM)
    p[4] = 0.01      # γI: Insulin-Aβ binding rate (1/(nM·hr))
    p[5] = 0.5       # δI: BBB transport rate (1/hr)
    p[6] = 100.0     # KBBB: BBB saturation constant (nM)
    
    # Amyloid-β parameters
    p[7] = 5.0       # αAβ: Basal Aβ production rate (nM/hr)
    p[8] = 1.5       # βAβ: IDE-mediated Aβ degradation rate (1/hr)
    p[9] = 30.0      # KAβ: Michaelis constant for Aβ-IDE (nM)
    p[10] = 2.0      # κ: Competitive inhibition strength
    p[11] = 0.1      # δAβ: Non-IDE Aβ clearance rate (1/hr)
    p[12] = 0.05     # μ: Inflammation effect on Aβ (nM⁻¹)
    p[13] = 0.5      # ξ: Trans-Golgi contribution
    p[14] = 1.0      # KTG: Trans-Golgi saturation
    p[15] = 0.8      # TG: Trans-Golgi activity
    
    # Tau parameters
    p[16] = 0.5      # kf: Tau phosphorylation rate (1/hr)
    p[17] = 0.3      # kr: Tau dephosphorylation rate (1/hr)
    p[18] = 100.0    # τ0: Total tau concentration (nM)
    p[19] = 2.0      # φ: GSK-3β activity coefficient
    p[20] = 3.0      # ψOG: O-GlcNAc effect on tau
    
    # Inflammation parameters
    p[21] = 0.2      # ρ: Aβ-induced inflammation rate (hr⁻¹)
    p[22] = 0.1      # σ: AGE-induced inflammation rate (nM⁻¹·hr⁻¹)
    p[23] = 0.5      # λInf: Inflammation resolution rate (hr⁻¹)
    p[24] = 20.0     # KInf: Half-saturation for Aβ inflammation (nM)
    
    # AGE parameters
    p[25] = 0.1      # νAGE: AGE formation rate (hr⁻¹)
    p[26] = 0.05     # δAGE: AGE clearance rate (hr⁻¹)
    p[27] = 1.2      # HG: Hyperglycemia factor (>1 = hyperglycemic)
    
    # O-GlcNAcylation parameters
    p[28] = 0.5      # kHBP: HBP pathway rate (hr⁻¹)
    p[29] = 5.0      # G: Intracellular glucose (mM)
    p[30] = 1.0      # GFAT: GFAT enzyme level
    p[31] = 2.0      # KGFAT: GFAT Michaelis constant
    p[32] = 0.3      # kOGA: OGA activity (hr⁻¹)
    p[33] = 1.0      # OGA: OGA enzyme level
    p[34] = 10.0     # KOG: OGA Michaelis constant (nM)
    
    # GLUT1 parameters
    p[35] = 1.0      # αGLUT1: GLUT1 production rate
    p[36] = 0.2      # δGLUT1: GLUT1 degradation rate (hr⁻¹)
    p[37] = 2.0      # θTX: TXNIP inhibition strength
    p[38] = 1.0      # TXNIP0: Basal TXNIP level
    p[39] = 3.0      # νTX: Akt inhibition of TXNIP
    
    # Insulin signaling parameters
    p[40] = 40.0     # KIS: Half-saturation for insulin signaling (nM)
    p[41] = 0.5      # η: Aβ effect on insulin signaling
    p[42] = 0.3      # ψInf: Inflammation effect on insulin signaling
    p[43] = 2.0      # χ: PTEN effect on insulin signaling
    p[44] = 1.0      # κAkt: Akt activation coefficient
    
    # Additional signaling parameters
    p[45] = 1.0      # GSK3β0: Basal GSK-3β activity
    p[46] = 3.0      # θGSK: Insulin inhibition of GSK-3β
    p[47] = 0.5      # PP2A: PP2A phosphatase level
    p[48] = 0.5      # PP1: PP1 phosphatase level
    p[49] = 50.0     # Kτ: Half-saturation for phosphatase (nM)
    
    # PTEN-tau feedback
    p[50] = 1.0      # PTEN0: Basal PTEN activity
    p[51] = 0.5      # ωτ: Tau inhibition of PTEN
    p[52] = 80.0     # Kτ0: Half-saturation for tau-PTEN (nM)
    
    # IDE level
    p[53] = 1.0      # IDE: Insulin-degrading enzyme level
    
    return p
end

# Define initial conditions for healthy state
function healthy_initial_conditions()
    u0 = zeros(7)
    u0[1] = 50.0     # I: Insulin (nM) - normal physiological range
    u0[2] = 5.0      # Aβ: Amyloid-β (nM) - low in healthy state
    u0[3] = 10.0     # τp: Phosphorylated tau (nM) - low in healthy state
    u0[4] = 0.1      # Inf: Inflammation (normalized) - minimal
    u0[5] = 0.5      # AGE: Advanced glycation end products (normalized)
    u0[6] = 20.0     # OGlcNAc: O-GlcNAcylation level (nM)
    u0[7] = 1.0      # GLUT1: GLUT1 expression (normalized)
    return u0
end

# Define initial conditions for pre-diabetic/insulin resistant state
function insulin_resistant_initial_conditions()
    u0 = zeros(7)
    u0[1] = 80.0     # I: Elevated insulin (hyperinsulinemia)
    u0[2] = 8.0      # Aβ: Slightly elevated
    u0[3] = 15.0     # τp: Slightly elevated
    u0[4] = 0.3      # Inf: Mild inflammation
    u0[5] = 1.0      # AGE: Elevated
    u0[6] = 15.0     # OGlcNAc: Reduced
    u0[7] = 0.8      # GLUT1: Slightly reduced
    return u0
end

# Run simulation
function run_simulation(u0=healthy_initial_conditions(), 
                        p=default_parameters(), 
                        tspan=(0.0, 500.0))
    """
    Run the AD metabolism model simulation
    
    Args:
        u0: Initial conditions vector
        p: Parameter vector
        tspan: Time span for simulation (hours)
    
    Returns:
        solution object from DifferentialEquations.jl
    """
    
    # Define the ODE problem
    prob = ODEProblem(ad_metabolism_model!, u0, tspan, p)
    
    # Solve using automatic algorithm selection
    # Tsit5() is good for non-stiff problems
    # For stiff problems, could use Rodas5()
    sol = solve(prob, Tsit5(), saveat=1.0)
    
    return sol
end

# Plotting function
function plot_results(sol, title="")
    """
    Create comprehensive plots of the simulation results
    """
    
    # Extract time points
    t = sol.t
    
    # Create subplots
    p1 = plot(t, sol[1,:], label="Insulin", xlabel="Time (hrs)", 
              ylabel="Concentration (nM)", title="Insulin Dynamics")
    
    p2 = plot(t, sol[2,:], label="Aβ", xlabel="Time (hrs)", 
              ylabel="Concentration (nM)", title="Amyloid-β Dynamics", 
              color=:red)
    
    p3 = plot(t, sol[3,:], label="p-Tau", xlabel="Time (hrs)", 
              ylabel="Concentration (nM)", title="Phosphorylated Tau", 
              color=:green)
    
    p4 = plot(t, sol[4,:], label="Inflammation", xlabel="Time (hrs)", 
              ylabel="Level (normalized)", title="Inflammation", 
              color=:orange)
    
    p5 = plot(t, sol[5,:], label="AGEs", xlabel="Time (hrs)", 
              ylabel="Level (normalized)", title="Advanced Glycation End Products", 
              color=:purple)
    
    p6 = plot(t, sol[6,:], label="O-GlcNAc", xlabel="Time (hrs)", 
              ylabel="Concentration (nM)", title="O-GlcNAcylation", 
              color=:brown)
    
    p7 = plot(t, sol[7,:], label="GLUT1", xlabel="Time (hrs)", 
              ylabel="Expression (normalized)", title="GLUT1 Transporter", 
              color=:pink)
    
    # Combine plots
    plot(p1, p2, p3, p4, p5, p6, p7, layout=(4,2), size=(1600, 1000), plot_title=title)
end

# Parameter sensitivity analysis function
function parameter_sensitivity(param_idx, param_range, u0=healthy_initial_conditions())
    """
    Analyze sensitivity of the model to a specific parameter
    
    Args:
        param_idx: Index of parameter to vary
        param_range: Range of parameter values to test
        u0: Initial conditions
    
    Returns:
        Dictionary with final values for each state variable
    """
    
    p_base = default_parameters()
    results = Dict()
    
    for val in param_range
        p = copy(p_base)
        p[param_idx] = val
        
        sol = run_simulation(u0, p, (0.0, 200.0))
        
        # Store final values
        results[val] = sol[:,end]
    end
    
    return results
end

# Example usage
function example()
    println("Running AD Energy Metabolism Model...")
    
    # Run healthy simulation
    println("\n1. Simulating healthy state...")
    sol_healthy = run_simulation()
    
    # Run insulin resistant simulation
    println("2. Simulating insulin resistant state...")
    sol_resistant = run_simulation(insulin_resistant_initial_conditions())
    
    # Create comparison plots
    println("3. Creating plots...")
    p_healthy = plot_results(sol_healthy, "Healthy")
    p_resistant = plot_results(sol_resistant, "Insulin Resistant")
    
    # Display plots
    display(p_healthy)
    display(p_resistant)
    
    # Example sensitivity analysis
    println("\n4. Running sensitivity analysis for IDE level...")
    ide_range = 0.5:0.25:2.0
    sensitivity = parameter_sensitivity(53, ide_range)
    
    # Plot sensitivity results for Aβ
    aβ_sensitivity = [sensitivity[val][2] for val in ide_range]
    p_sens = plot(ide_range, aβ_sensitivity, 
                  xlabel="IDE Level (normalized)", 
                  ylabel="Final Aβ (nM)",
                  title="Aβ Sensitivity to IDE",
                  marker=:circle)
    display(p_sens)
    
    println("\nSimulation complete!")
    
    # return sol_healthy, sol_resistant
end

end