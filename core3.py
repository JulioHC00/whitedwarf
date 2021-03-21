'''
USED TO SOLVE THE CORE OF A WHITE DWARF

core.solve: Solves the core, for help use help(core.solve)
'''


import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
import scipy.constants as sc

def solve(rho, T_c, Y_e, rho_r = -1, graphs = False, R_r = 6.9634e8, r_o = 0.0001, x_max = -1, density_event = False, messages = True, min_density = 1e-15, solver = "RK23"):

    '''
    
    Description
    -----------
    Solves the core of a White Dwarf using scipy.integrate.solve_ivp
    
    Stops when temperature of the core equals fermi temperature.
    Works under the following assumptions:
    
    - Isothermal core
    - Constant element distribution with free electrons per nucleon equal to Y_e
    - Full degeneracy all through the core
    - Only source of pressure is electron degeneracy pressure
    - Classical fermi momentum sqrt(2*m_e*E_fermi). The calculation doesn't work in the high relativistic regime, it deviates.
    
    Definition of reduced variables is:
    
    reduced density = density/rho_r
    reduced radius = r/R_r
    reduced mass = M/(4/3*pi*R_r^3*rho_r)
    reduced temperature = T/T_c
    reduced pressure = P/P_c [Where P_c is the degenerate electron pressure at rho_r density]
    
    Parameters
    ----------
    
    param rho: float, central density of the star in kg/m^3.
    T_c: float
        core temperature in K.
    rho_r: float, optional
        density in kg/m^3 used to reduce the density variable for integration.
        Default is equal to rho.
    graph: boolean, optional
        whether to print graphs or not. False by default.
    R_r: float, optional
        radius in m used to reduce the radius variable for integration.
        Default is the Sun radius.
    r_o: float, optional
        educed radius from where to start integrating.
        Default is 1e-4.
    x_max: float, optional
        reduced radius upper limit for integration.
        Default is equivalent to 2 solar radii.
    density_event: boolean, optional
        Whether to stop at a defined minimum density
    min_density: float, optional
        Minimum reduced density where to stop is density_event=True
        Default is 1e-15
    solver: string, optional
        Which method to use for solving the core.
        Available methods are those available for scipy.integrate.solve_ivp
        Default is RK23
    messages: boolean, optional
        Whether to print messages about the result of integrations.
        When the function is to be used in a loop it's recommended to set to false.
        Default is True
        
    Returns
    -------
    
    core, reduced_core corresponding to SI units and reduced (dimensionless) variables.
    The following fields are define for each
    
    mass = Mass of the core 
    density = Density of the core
    temperature = Temperature of the core
    pressure = Pressure of the core
        Calculated with differential equation
    pressure_sm = Pressure of the core
        Calculated as fully degenerate electron pressure
    '''
    
    if rho_r == -1:
        rho_r = rho
    
    rho_core_sun = 1.62e5
    rho_mean_sun = 1406.3134
    R_sun = 6.9634*1e8
    M_sun = 2*1e30
    P_c = ((2.*sc.pi*sc.hbar**2)/(5*sc.m_e))*(4*sc.pi/3)**(-2./3)*2**(-2./3)*(rho*Y_e/sc.m_p)**(5./3)
    rho_o = (sc.m_e*sc.c/sc.hbar)**(3)*sc.m_p/(3*sc.pi**2*Y_e)
    q_o = rho_o/rho_r
    
    if x_max == -1:
        x_max = 3e10/R_r

    def change_to_envelope(x,variables):
        q, M, t = variables
        tf = (sc.hbar**2/(2*sc.m_e*sc.k)*(((3*sc.pi**2*rho_r*Y_e)/(sc.m_p))*q)**(2./3))/(T_c)
        if t>=tf:
            envelope = 0
        else:
            envelope = 1
        return envelope
    change_to_envelope.terminal = True

    def minimum_density(x,variables):
        q, M, t = variables 
        if q<=min_density:
            terminate = 0
        elif np.isnan(q):
            terminate = 0
        else:
            terminate = 1
        return terminate
    minimum_density.terminal = True

    def gamma(q):
        y = q/q_o
        gam = y**(2/3)/(3*(1+y**(2/3))**(1/2))
        return gam

    def core(x,variables):
        q, M, t= variables
        C = 4*sc.G*R_r**2*sc.m_p*sc.pi*rho_r/(3*Y_e*sc.c**2*sc.m_e)
        derivatives = [-C*M*q/(gamma(q)*x**2),
                      3*q*x**2,
                      0]
        return derivatives  
    
    def pressure(q, P):
        dpdq = Y_e*sc.c**2*gamma(q)*sc.m_e*rho_r/(P_c*sc.m_p)
        return dpdq
        
    if density_event:
        c_l = solve_ivp(core,[r_o,x_max],[rho/rho_r,0,1], events = [change_to_envelope,minimum_density], method = solver)
    else:
        c_l = solve_ivp(core,[r_o,x_max],[rho/rho_r,0,1], events = change_to_envelope, method = solver)

    class core:
        mass = c_l.y[1]*(4./3)*sc.pi*(R_r)**3*rho_r
        mass = mass[~np.isnan(mass)]
        radius = c_l.t*R_r
        radius = radius[0:len(mass)]
        density = c_l.y[0]*rho_r
        density = density[~np.isnan(density)]
        temperature = c_l.y[2]*T_c
        temperature = temperature[0:len(mass)]
    class reduced_core:
        
        mass = c_l.y[1]
        mass = mass[~np.isnan(mass)]
        radius = c_l.t
        radius = radius[0:len(mass)]
        density = c_l.y[0]
        density = density[~np.isnan(density)]
        temperature = c_l.y[2]
        temperature = temperature[0:len(mass)]
    
    if messages:
        print("Core:")
        print(c_l.message)
    
    pressure = solve_ivp(pressure,[reduced_core.density[0],reduced_core.density[-1]], [1], method = solver, t_eval = reduced_core.density)
    
    if messages:
        print("Pressure:")
        print(pressure.message)
    
    core.pressure = pressure.y[0]*P_c
    reduced_core.pressure = pressure.y[0]
    
    core.pressure_sm = ((2.*sc.pi*sc.hbar**2)/(5*sc.m_e))*(4*sc.pi/3)**(-2./3)*2**(-2./3)*(core.density*Y_e/sc.m_p)**(5./3)
    reduced_core.pressure_sm = core.pressure/P_c

    if graphs:
        fig, ax = plt.subplots(2,2, figsize=(13,10), dpi=100)
        ax[0,0].plot(core.radius/R_sun, core.mass/M_sun)
        ax[0,0].set_xlabel('Core radius (R⊙)')
        ax[0,0].set_ylabel('Core mass (M⊙)')
        ax[0,0].set_title('Core mass (M⊙)')
        ax[0,0].grid()
        
        ax[0,1].plot(core.radius/R_sun, core.density)
        ax[0,1].set_xlabel('Core radius (R⊙)')
        ax[0,1].set_ylabel('Core density (kg/m^3)')
        ax[0,1].set_title('Core density (kg/m^2)')
        ax[0,1].grid()
        
        ax[1,0].plot(core.radius/R_sun,core.pressure, label = 'dP/drho method')
        ax[1,0].plot(core.radius/R_sun,((2.*sc.pi*sc.hbar**2)/(5*sc.m_e))*(4*sc.pi/3)**(-2./3)*2**(-2./3)*(core.density*Y_e/sc.m_p)**(5./3), label = 'Stat. Mec. Notes', linestyle = 'dotted', color ='orange')
        ax[1,0].set_xlabel('Core radius (R⊙)')
        ax[1,0].hlines((core.density[-1]/(sc.m_p))*sc.k*core.temperature[-1], core.radius[0]/R_sun, core.radius[-1]/R_sun, color = 'green', linestyle = '-.', label = 'PV=NkT')
        ax[1,0].set_ylabel('Log Core pressure (Pa)')
        ax[1,0].set_title(' Log Core pressure (Pa)')
        ax[1,0].grid()
        ax[1,0].legend()
        ax[1,0].set_yscale('log')
        
        ax[1,1].plot(core.radius/R_sun, core.temperature, label = 'Core temperature', linestyle = 'dotted', color = 'orange')
        ax[1,1].plot(core.radius/R_sun, (sc.hbar**2/(2*sc.m_e*sc.k)*(((3*sc.pi**2*rho_r*Y_e)/(sc.m_p))*reduced_core.density)**(2./3)), label = 'Fermi Temperature')
        ax[1,1].set_ylabel('Temperature (K)')
        ax[1,1].set_xlabel('Core radius (R⊙)')
        ax[1,1].set_title('Core temperature (K)')
        ax[1,1].grid()
        ax[1,1].legend()
    return core, reduced_core
