import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
import scipy.constants as sc
import core4
import scipy.interpolate

rho_core_sun = 1.62e5
rho_mean_sun = 1406.3134
R_sun = 6.9634*1e8
M_sun = 2*1e30
L_sun = 3.828e26
halfsolardens = 1.9677e9
m_u = 1.6605390666e-27

def solve(rho_core, T_core, Y_e, X, Y, Z, graphs=False, message = True, x_max=-1, rho_r=1e4, R_r = 1e4, P_r = -1, T_r = -1, density_cutoff = -1, l = 1, density = True, solver = 'RK23', r_tol_core = 1e-3, a_tol_core = 1e-6, r_tol_envelope = 1e-3, a_tol_envelope = 1e-6):
    cor, re_cor = core4.solve(rho_core, T_core, Y_e, messages = message, r_tol = r_tol_core, a_tol = a_tol_core)

    rho_o = cor.density[-1]
    m_o = cor.mass[-1]
    P_o = cor.pressure[-1]
    R_o = cor.radius[-1]
    T_o = cor.temperature[-1]

    mu = 2/(1+3*X+0.5*Y)
    mu_e = X+2*Y+2*Z
    Y_e_env = 1/mu_e
    
    if P_r == -1:
        P_r = P_o
    if T_r == -1:
        T_r = T_o
    if x_max == -1:
        x_max = 1e9/R_r

    q_o = rho_o/rho_r
    M_o = m_o/((4/3)*sc.pi*R_r**3*rho_r)
    p_o = P_o/P_r
    x_o = R_o/R_r
    t_o = T_o/T_r
    
    kappa_o = 4.34e27*Z*(1+X)
    L = (32/(3*8.5))*sc.sigma*(4*sc.pi*sc.G*m_o/kappa_o)*mu*m_u/(sc.k)*T_o**(6.5)/(rho_o**2)
    
    if density_cutoff == -1:
        density_cutoff = 1/rho_r
        
    def density_calc(M,t):
        m = 4/3*sc.pi*R_r**3*rho_r*M
        rho_total = ((2/8.5)*(16*sc.sigma/3)*(4*sc.pi*sc.G*(m+m_o)/(kappa_o*L))*(mu*m_u/sc.k))**(1/2)*((t+t_o)*T_r)**(3.25)
        rho = rho_total -rho_o
        return rho
 
    def min_density(x, variables):
        M, t = variables
        
        q = density_calc(M,t)/rho_r

        if q_o+q <= density_cutoff:
            end = 0
        else:
            end = 1
        return end
    min_density.terminal = True

    def opacity(q, t):
        kappa = 4.34e27*Z*(1+X)*rho_r*(q+q_o)*(T_r*t)**(-3.5)
        return kappa
    

    def envelope_equations(x, variables):
        M, t = variables
        q = density_calc(M,t)/rho_r
        kappa = opacity(q+q_o, t+t_o)
        Ct = 3*kappa*L*rho_r/(64*R_r*T_r**4*sc.pi*sc.sigma)
        derivatives = [3*(q+q_o)*x**2,
                       -Ct*(q + q_o)/((t + t_o)**3*(x + x_o)**2)]
        return derivatives
    
    if density:
        env = solve_ivp(envelope_equations, [0,x_max], [0,0], method = solver, events = min_density, rtol = r_tol_envelope, atol = a_tol_envelope)
    elif ~density:
        env = solve_ivp(envelope_equations, [0,x_max], [0,0], method = solver, rtol = r_tol_envelope, atol = a_tol_envelope)
    
    if message:
        print('Envelope:', env.message)

    class envelope:
        mass = m_o + env.y[0]*(4/3)*sc.pi*R_r**3*rho_r
        mass = mass[~np.isnan(mass)]
        density = rho_o + density_calc(env.y[0],env.y[1])
        density = density[0:len(mass)]
        temperature = T_o+env.y[1]*T_r
        temperature = temperature[0:len(mass)]
        radius = R_o + env.t*R_r
        radius = radius[0:len(mass)]
        luminosity = L
        surface_temp = (L/(4*sc.pi*((x_o+env.t[-1])*R_r)**2*sc.sigma))**(1/4)
    class del_envelope:
        mass = env.y[0]*(4/3)*sc.pi*R_r**3*rho_r
        mass = mass[~np.isnan(mass)]
        density = density_calc(env.y[0],env.y[1])
        density = density[0:len(mass)]
        temperature = env.y[1]*T_r
        temperature = temperature[0:len(mass)]
        radius = env.t*R_r
        radius = radius[0:len(mass)]
    class reduced_del_envelope:
        mass = env.y[0]
        mass = mass[~np.isnan(mass)]
        density = density_calc(env.y[0], env.y[1])
        density = density[0:len(mass)]
        temperature = env.y[1]
        temperature = temperature[0:len(mass)]
        radius = env.t
        radius = radius[0:len(mass)]
    
    inter_q = scipy.interpolate.interp1d(reduced_del_envelope.radius, reduced_del_envelope.density)
    inter_t = scipy.interpolate.interp1d(reduced_del_envelope.radius, reduced_del_envelope.temperature)
    inter_M = scipy.interpolate.interp1d(reduced_del_envelope.radius, reduced_del_envelope.mass)
    
    def pressure(x, p):
        M = inter_M(x)
        t = inter_t(x)
        q = density_calc(M,t)/rho_r
        C = 4*sc.G*R_r**2*sc.pi*rho_r**2/(3*P_r)
        dpdx = -C*(M+M_o)*(q+q_o)/(x+x_o)**2
        return dpdx
    
    pressure = solve_ivp(pressure,[reduced_del_envelope.radius[0],reduced_del_envelope.radius[-1]], [0], t_eval = reduced_del_envelope.radius, method = solver)
    
    envelope.pressure = P_o+pressure.y[0]*P_r
    del_envelope.pressure = pressure.y[0]*P_r
    reduced_del_envelope.pressure = pressure.y[0]
    
    if graphs:

        fig, ax = plt.subplots(4,2, dpi = 150, figsize=(15,25))

        ax[0,0].plot(del_envelope.radius, del_envelope.density, color = 'orange')
        ax[0,0].set_xlabel('Envelope radius (R-R_core) [m]')
        ax[0,0].set_ylabel('Envelope change in density (rho-rho_interface) [kg/m^3]')
        ax[0,0].set_title('Incremental density of the envelope')
        ax[0,0].grid()

        ax[0,1].plot(del_envelope.radius, del_envelope.mass, color = 'orange')
        ax[0,1].set_xlabel('Envelope radius (R-R_core) [m]')
        ax[0,1].set_ylabel('Envelope change in mass (M-M_core) [kg]')
        ax[0,1].set_title('Incremental mass of the envelope')
        ax[0,1].grid()

        ax[1,0].plot(del_envelope.radius, del_envelope.temperature, color = 'orange')
        ax[1,0].set_xlabel('Envelope radius (R-R_core) [m]')
        ax[1,0].set_ylabel('Envelope change in temperature  (T-T_core) [K]')
        ax[1,0].set_title('Incremental temperature of the envelope')
        ax[1,0].grid()

        ax[1,1].plot(envelope.radius/R_sun, envelope.density, color = 'green')
        ax[1,1].set_xlabel('Total radius [R⊙]')
        ax[1,1].set_ylabel('Envelope density [kg/m^3]')
        ax[1,1].set_title('Density of the envelope')
        ax[1,1].grid()

        ax[2,0].plot(envelope.radius/R_sun, envelope.mass/M_sun, color = 'green')
        ax[2,0].set_xlabel('Total radius [R⊙]')
        ax[2,0].set_ylabel('Envelope mass [M⊙]')
        ax[2,0].set_title('Total mass')
        ax[2,0].set_yticks(np.arange(0,1,0.1))
        ax[2,0].grid()

        ax[2,1].plot(envelope.radius/R_sun, envelope.temperature, color = 'green')
        ax[2,1].set_xlabel('Total radius [R⊙]')
        ax[2,1].set_ylabel('Envelope temperature [K]')
        ax[2,1].set_title('Temperature of the envelope')
        ax[2,1].grid()
        
        ax[3,0].plot(envelope.radius/R_sun, envelope.pressure, color = 'green', label = 'dPdr')
        ax[3,0].plot(envelope.radius/R_sun, (envelope.density/sc.m_p)*sc.k*envelope.temperature, label = 'P=rho*k*T/m_p', color='blue', linestyle='dotted')
        ax[3,0].set_xlabel('Total radius [R⊙]')
        ax[3,0].set_ylabel('Envelope pressure [Pa]')
        ax[3,0].set_title('Pressure of the envelope')
        ax[3,0].grid()
        ax[3,0].legend()

    print(density_calc(0,0)+rho_o, cor.density[-1])
    
    return envelope, del_envelope, reduced_del_envelope, cor, re_cor
