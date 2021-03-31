import core
import envelope
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.constants as sc

rho_core_sun = 1.62e5
rho_mean_sun = 1406.3134
R_sun = 6.9634*1e8
M_sun = 2*1e30
L_sun = 3.828e26
halfsolardens = 1.9677e9
m_u = 1.6605390666e-27
million_years = 1e6*365*24*3600

def calculate(M, T_o, R, Y_e_core, X, Y, Z, solver = 'RK23', r_tol = 1e-3, a_tol = 1e-6, graphs = False, t_max = 100):
    kappa_o = 4.34e23*Z*(1+X)
    mu_e = 1/Y_e_core
    mu = 2/(1+3*X+0.5*Y)
    
    Cv = (3/2)*sc.k
    
    def equations(t, T):
        rho_c = 6e-6*mu_e*T**(3/2)
        L = (32/(3*8.5))*sc.sigma*(4*sc.pi*sc.G*M/kappa_o)*mu*m_u/(sc.k)*T**(6.5)/(rho_c**2)
        
        dTdt = -1000*million_years*L*mu*m_u/(Cv*M)
        return dTdt
    
    cool = solve_ivp(equations, [0,t_max], [T_o], method = solver, rtol = r_tol, atol = a_tol)
    
    class evolution:
        time = cool.t*million_years*1000
        core_temperature = cool.y[0] 
        luminosity = (32/(3*8.5))*sc.sigma*(4*sc.pi*sc.G*M/kappa_o)*mu*m_u/(sc.k)*core_temperature**(3.5)/((6e-6*mu_e)**2)
        surface_temperature = (luminosity/(4*sc.pi*R**2*sc.sigma))**(1/4)
        
    if graphs:
        fig, ax = plt.subplots(3,1, dpi =200, figsize = (5,30))
        ax[0].plot(evolution.time/(1000*million_years), evolution.core_temperature)
        ax[0].set_xlabel('Billion years')
        ax[0].set_ylabel('Core temperature [K]')
        ax[0].grid()
        
        ax[1].plot(evolution.time/(1000*million_years), evolution.surface_temperature)
        ax[1].set_xlabel('Billion years')
        ax[1].set_ylabel('Surface temperature [K]')
        ax[1].grid()
        
        ax[2].plot(evolution.time/(1000*million_years), evolution.luminosity/L_sun)
        ax[2].set_xlabel('Billion years')
        ax[2].set_ylabel('Solar luminosities')
        ax[2].grid()
        
    
    return evolution

def full_calculate(rho_core, T_core, Y_e_core, X, Y, Z, solver = 'RK23', r_tol = 1e-3, a_tol = 1e-6, graphs = False, t_max = 100):

    env, cor = envelope.solve(rho_core,T_core, Y_e_core, X, Y, Z, solver = solver, r_tol_core = r_tol, r_tol_envelope = r_tol, a_tol_core = a_tol, a_tol_envelope = a_tol, message = False)
    
    evolution = calculate(cor.mass[-1], cor.temperature[-1], env.radius[-1], Y_e_core, X, Y, Z, graphs = graphs, solver = solver, t_max = t_max)
    return evolution
