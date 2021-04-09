'''
SOLVES THE ENVELOPE AND CORE OF A WHITE DWARF.
NEEDS THE core MODULE, FOR ASSUMPTIONS AND INSTRUCTIONS OF JUST THE CORE SEE core
The envelope is solved according to the following assumptions:
- Kramer's opacity law
- Radiative diffusion as the form of energy transport
- Constant luminosity
- Ideal gas equation of state
- Constant gas composition
- Fully ionised gas in the envelope
See envelope.solve for more details
'''


import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
import scipy.constants as sc
import core
import scipy.interpolate

rho_core_sun = 1.62e5
rho_mean_sun = 1406.3134
R_sun = 6.9634 * 1e8
M_sun = 2 * 1e30
L_sun = 3.828e26
halfsolardens = 1.9677e9
m_u = 1.6605390666e-27


def lum(
        rho,
        T,
        Y_e,
        X,
        Y,
        Z,
        message=False,
        r_tol_core=1e-3,
        a_tol_core=1e-6,
        core_solver='RK23'):
    '''
    Description
    -----------

    USED IF ONLY THE LUMINOSITY AND SURFACE TEMPERATURE ARE NEEDED
    FASTER THAN SOLVING THE ENVELOPE BUT NOT AS ACCURATE
    RADIUS TAKEN TO BE CORE RADIUS NOT INCLUDING THE ENVELOPE

    Parameters
    ----------

    rho: float
        Density at r=0 for the core in kg/m^2
    T: float
        Temperature of the isothermal core
    Y_e: float
        Free electrons per nucleon FOR THE CORE
    X: float
        Hydrogen mass fraction FOR THE ENVELOPE
    Y: float
        Helium mass fraction FOR THE ENVELOPE
    Z: float
        Metals mass fraction FOR THE ENVELOPE
    message: boolean, optional
        Whether to print messages about the procces status
        Set to false if the function is going to be used in a loop
        Default is True
    r_tol_core: float, optional
        Maximum relative error for the core
        For detailed information refer to documentation of 'solve_ivp' from scipy
        Default is 1e-3
    a_tol_core: float, optional
        Maximum absolute error for the core
        For detailed information refer to documentation of 'solve_ivp' from scipy
        Default is 1e-6
    core_solver: string, optional
        Which method to use for solve_ivp in the core
        Default is 'RK23'

    Returns
    -------

    luminosity: class
        Containing the following information

        luminosity.value | Value of the luminosity in W
        luminosity.temperature | Surface temperature in K calculated with the radius of the core

    '''

    # SOLVE CORE
    cor, re_cor = core.solve(rho, T, Y_e, messages=message, r_tol=r_tol_core,
                             a_tol=a_tol_core, X=X, Y=Y, Z=Z, solver=core_solver)

    # STORE VALUES OF THE CORE
    rho_o = cor.density[-1]
    m_o = cor.mass[-1]
    R_o = cor.radius[-1]
    T_o = cor.temperature[-1]

    # CALCULATE OPACITY, LUMINOSITY AND SURFACE TEMPERATURE
    mu = 2 / (1 + 3 * X + 0.5 * Y)
    kappa_o = 4.34e23 * Z * (1 + X)
    L = (32 / (3 * 8.5)) * sc.sigma * (4 * sc.pi * sc.G * m_o /
                                       kappa_o) * mu * m_u / (sc.k) * T_o**(6.5) / (rho_o**2)
    surface_temp = (L / (4 * sc.pi * R_o**2 * sc.sigma))**(1 / 4)

    # STORE VALUES
    class luminosity:
        value = L
        temperature = surface_temp
    return luminosity


def solve(
        rho_core,
        T_core,
        Y_e,
        X,
        Y,
        Z,
        graphs=False,
        message=False,
        x_max=-1,
        rho_r=1e4,
        R_r=1e4,
        P_r=-1,
        T_r=-1,
        density=True,
        density_cutoff=-1,
        solver='RK23',
        core_solver='RK23',
        r_tol_core=1e-3,
        a_tol_core=1e-6,
        r_tol_envelope=1e-3,
        a_tol_envelope=1e-6,
        full_return=False):
    '''
    Description
    -----------

    SOLVES THE ENVELOPE RETURNING SEVERAL OBJECTS.
    WORKS UNDER THE ASSUMPTIONS LISTED IN THE MODULE DESCRIPTION.

    The reduced variables constants (R_r, rho_r, T_r, P_r) are used to define the following reduced variables:

        q_o = rho_o/rho_r             | Reduced density at the interface core-envelope
        q   = rho/rho_r               | Reduced change in density from the core-envelope interface
        M_o = m_o/(4*pi*R_r**3*rho_r) | Reduced mass at the interface core-envelope
        M   = m/(4*pi*R_r**3*rho_r)   | Reduced change in mass from the core-envelope interface
        x_o = R_o/R_r                 | Reduced radius at the interface core-envelope
        x   = R/R_r                   | Reduced radius increment from the core-envelope interface
        t_o = T_o/T_r                 | Reduced temperature at the interface core-envelope
        t   = t/T_r                   | Reduced temperature increment from the core-envelope interface
        p_o = P_o/P_r                 | Reducesd pressure at the interface core-envelope
        p   = P/p                     | Reduced pressure increment from the core-envelope interface

    Such that the total reduced quantities at some radius can be written as:

        q_total = q + q_o
        M_total = M + M_o
        x_total = x + x_o
        t_total = t + t_o
        p_total = p + p_o

    Parameters
    ----------

    rho_core: float
        Density at r=0 for the core in kg/m^3
    T_core: float
        Temperature of the isothermal core
    Y_e: float
        Free electrons per nucleon FOR THE CORE
    X: float
        Hydrogen mass fraction FOR THE ENVELOPE
    Y: float
        Helium mass fraction FOR THE ENVELOPE
    Z: float
        Metals mass fraction FOR THE ENVELOPE
    graphs: boolean, optional
        Whether to produce some default plots
        Default is False
    message: boolean, optional
        Whether to print messages about the procces status
        Set to false if the function is going to be used in a loop
        Default is True
    x_max: float, optional
        Maximum reduced radius for integration
        Default is equivalent to 1e9 meters
    rho_r: float, optional
        Value used to reduce the density
        Default is 1e4 kg/m^3
    R_r: float, optional
        Value used to reduce the radius
        Default is 1e4 meters
    P_r: float, optional
        Value used to reduce the pressure
        Default is set equal to the pressure at the end of the core
    T_r: float, optional
        Value used to reduce the temperature
        Default is set equal to the core temperature
    density: boolean, optional
        Whether to use a minimum density condition to stop the integration
        Default is True
    density_cutoff: float, optional
        If density = True, at which minimum density to stop integration
        Default is equal to 1/rho_r**3
    solver: string, optional
        Which method to use for solve_ivp in the envelope
        Default is 'RK23'
    core_solver: string, optional
        Which method to use for solve_ivp in the core
        Default is 'RK23'
    r_tol_core: float, optional
        Maximum relative error for the core
        For detailed information refer to documentation of 'solve_ivp' from scipy
        Default is 1e-3
    a_tol_core: float, optional
        Maximum absolute error for the core
        For detailed information refer to documentation of 'solve_ivp' from scipy
        Default is 1e-6
    r_tol_envelope: float, optional
        Same as r_tol_core but for the envelope
    a_tol_envelope: float, optional
        Same as a_tol_core but for the envelope
    full_return: boolean, optional
        If True the function returns envelope, del_envelope, reduced_del_envelope, cor, re_cor
        If False the function only return envelope, cor
        Default is False

    Returns
    -------

    envelope, del_envelope, reduced_del_envelope, cor, re_cor

    envelope: class
        Contain the values for the envelope at increasing radius according to:

        envelope.mass         | Mass in kg, array
        envelope.density      | Density in kg/m^3, array
        envelope.temperature  | Temperature in K, array
        envelope.pressure     | Pressure in Pa, array
        envelope.radius       | Radius in m, array
        envelope.luminosity   | Luminosity in W, float
        envelope.surface_temp | Surface temperature of the envelope in K, float

    del_envelope: class
        Same contents as the envelope class except luminosity and surface_temp and values are
        the increments of the specific variable with respect to the value of such variable at
        the core-envelope interface.

    reduced_del_envelope: class
        Same contents as del_envelope but the values are reduced according to the forms stated above

    cor: class
        Contains values for the core in increasing radius according to

        core.mass        | Mass in kg, array
        core.density     | Density in kg/m^3, array
        core.temperature | Temperature in K, array
        core.pressure    | Pressure in Pa, array
        core.radius      | Radius in m, array

    re_cor: class
        Same as the cor class but the values are reduced according to the procedure of core
    '''

    # SOLVE CORE
    cor, re_cor = core.solve(rho_core, T_core, Y_e, messages=message,
                             r_tol=r_tol_core, a_tol=a_tol_core, X=X, Y=Y, Z=Z, solver=core_solver)

    # STORE VALUES OF THE CORE
    rho_o = cor.density[-1]
    m_o = cor.mass[-1]
    P_o = cor.pressure[-1]
    R_o = cor.radius[-1]
    T_o = cor.temperature[-1]

    # DEFINE CONSTANTS TO REDUCE VARIABLES
    if P_r == -1:
        P_r = P_o
    if T_r == -1:
        T_r = T_o
    if x_max == -1:
        x_max = 1e9 / R_r

    # DEFINE DENSITY AT WHICH TO STOP INTEGRATING
    if density_cutoff == -1:
        density_cutoff = 1 / rho_r**3

    # REDUCED INTERFACE VALUES
    q_o = rho_o / rho_r
    M_o = m_o / ((4 / 3) * sc.pi * R_r**3 * rho_r)
    p_o = P_o / P_r
    x_o = R_o / R_r
    t_o = T_o / T_r

    # CALCULATE THE MEAN ION WEIGHT IN UNITS OF HYDROGEN MASS, NUCLEONS PER
    # FREE ELECTRON AND FREE ELECTRONS PER NUCLEON FOR THE ENVELOPE
    mu = 2 / (1 + 3 * X + 0.5 * Y)
    mu_e = ((1 / 2) * (1 + X))**(-1)
    Y_e_env = (1 / 2) * (1 + X)

    # OPACITY PROPORCIONALITY CONSTANT AND LUMINOSITY
    kappa_o = 4.34e23 * Z * (1 + X)
    L = (32 / (3 * 8.5)) * sc.sigma * (4 * sc.pi * sc.G * m_o /
                                       kappa_o) * mu * m_u / (sc.k) * T_o**(6.5) / (rho_o**2)

    # TO CALCULATE DENSITY
    def density_calc(M, t):
        m = 4 / 3 * sc.pi * R_r**3 * rho_r * M
        rho = ((2 / 8.5) * (16 * sc.sigma / 3) * (4 * sc.pi * sc.G * (m + m_o) /
                                                        (kappa_o * L)) * (mu * m_u / sc.k))**(1 / 2) * ((t + t_o) * T_r)**(3.25)
        return rho

    # TO CALCULATE OPACITY
    def opacity(q, t):
        kappa = kappa_o * rho_r * q * (T_r * (t + t_o))**(-3.5)
        return kappa

    # EVENT TO STOP AT MINIMUM DENSITY
    def min_density(x, variables):
        M, t = variables

        q = density_calc(M, t) / rho_r

        if q <= density_cutoff:
            end = 0
        else:
            end = 1
        return end
    min_density.terminal = True

    # DIFERENTIAL EQUATIONS OF THE ENVELOPE
    def envelope_equations(x, variables):
        M, t = variables
        q = density_calc(M, t) / rho_r
        kappa = opacity(q, t)
        Ct = 3 * kappa * L * rho_r / (64 * R_r * T_r**4 * sc.pi * sc.sigma)
        derivatives = [3 * q * (x + x_o)**2,
                       -Ct * q / ((t + t_o)**3 * (x + x_o)**2)]
        return derivatives

    # SOLVE THE ENVELOPE
    if density:
        env = solve_ivp(envelope_equations,
            [0, x_max],
            [0, 0],
            method=solver,
            events=min_density,
            rtol=r_tol_envelope,
            atol=a_tol_envelope)
    elif ~density:
        env = solve_ivp(
            envelope_equations,
            [0, x_max],
            [0, 0],
            method=solver,
            rtol=r_tol_envelope,
            atol=a_tol_envelope)

    if message:
        print('Envelope:')
        print(env.message)

    # STORE DATA
    class envelope:
        mass = m_o + env.y[0] * (4 / 3) * sc.pi * R_r**3 * rho_r
        mass = mass[~np.isnan(mass)]
        density = density_calc(env.y[0], env.y[1])
        density = density[0:len(mass)]
        temperature = T_o + env.y[1] * T_r
        temperature = temperature[0:len(mass)]
        radius = R_o + env.t * R_r
        radius = radius[0:len(mass)]
        luminosity = L
        surface_temp = (
            L / (4 * sc.pi * ((x_o + env.t[-1]) * R_r)**2 * sc.sigma))**(1 / 4)

    class del_envelope:
        mass = env.y[0] * (4 / 3) * sc.pi * R_r**3 * rho_r
        mass = mass[~np.isnan(mass)]
        density = density_calc(env.y[0], env.y[1]) - rho_o
        density = density[0:len(mass)]
        temperature = env.y[1] * T_r
        temperature = temperature[0:len(mass)]
        radius = env.t * R_r
        radius = radius[0:len(mass)]

    class reduced_del_envelope:
        mass = env.y[0]
        mass = mass[~np.isnan(mass)]
        density = (density_calc(env.y[0], env.y[1]) - rho_o) / rho_r
        density = density[0:len(mass)]
        temperature = env.y[1]
        temperature = temperature[0:len(mass)]
        radius = env.t
        radius = radius[0:len(mass)]

    # EXTRAPOLATION FOR PRESSURE
    inter_q = scipy.interpolate.interp1d(
        reduced_del_envelope.radius,
        reduced_del_envelope.density)
    inter_t = scipy.interpolate.interp1d(
        reduced_del_envelope.radius,
        reduced_del_envelope.temperature)
    inter_M = scipy.interpolate.interp1d(
        reduced_del_envelope.radius,
        reduced_del_envelope.mass)

    # PRESSURE EQUATION
    def pressure(x, p):
        M = inter_M(x)
        t = inter_t(x)
        q = (density_calc(M, t) - rho_o) / rho_r
        C = 4 * sc.G * R_r**2 * sc.pi * rho_r**2 / (3 * P_r)
        dpdx = -C * (M + M_o) * (q + q_o) / (x + x_o)**2
        return dpdx

    # SOLVE AND STORE PRESSURE
    pressure = solve_ivp(pressure,
                         [reduced_del_envelope.radius[0], reduced_del_envelope.radius[-1]],
                         [0],
                         t_eval=reduced_del_envelope.radius,
                         method=solver)
    envelope.pressure = P_o + pressure.y[0] * P_r
    del_envelope.pressure = pressure.y[0] * P_r
    reduced_del_envelope.pressure = pressure.y[0]

    # PLOTS
    if graphs:

        fig, ax = plt.subplots(4, 2, dpi=150, figsize=(15, 25))

        ax[0, 0].plot(del_envelope.radius,
                      del_envelope.density, color='orange')
        ax[0, 0].set_xlabel('Envelope radius (R-R_core) [m]')
        ax[0, 0].set_ylabel(
            'Envelope change in density (rho-rho_interface) [kg/m^3]')
        ax[0, 0].set_title('Incremental density of the envelope')
        ax[0, 0].grid()

        ax[0, 1].plot(del_envelope.radius, del_envelope.mass, color='orange')
        ax[0, 1].set_xlabel('Envelope radius (R-R_core) [m]')
        ax[0, 1].set_ylabel('Envelope change in mass (M-M_core) [kg]')
        ax[0, 1].set_title('Incremental mass of the envelope')
        ax[0, 1].grid()

        ax[1, 0].plot(del_envelope.radius,
                      del_envelope.temperature, color='orange')
        ax[1, 0].set_xlabel('Envelope radius (R-R_core) [m]')
        ax[1, 0].set_ylabel('Envelope change in temperature  (T-T_core) [K]')
        ax[1, 0].set_title('Incremental temperature of the envelope')
        ax[1, 0].grid()

        ax[1, 1].plot(envelope.radius / R_sun, envelope.density, color='green')
        ax[1, 1].set_xlabel('Total radius [R⊙]')
        ax[1, 1].set_ylabel('Envelope density [kg/m^3]')
        ax[1, 1].set_title('Density of the envelope')
        ax[1, 1].grid()

        ax[2, 0].plot(envelope.radius / R_sun,
                      envelope.mass / M_sun, color='green')
        ax[2, 0].set_xlabel('Total radius [R⊙]')
        ax[2, 0].set_ylabel('Envelope mass [M⊙]')
        ax[2, 0].set_title('Total mass')
        ax[2, 0].set_yticks(np.arange(0, 1, 0.1))
        ax[2, 0].grid()

        ax[2, 1].plot(envelope.radius /
                      R_sun, envelope.temperature, color='green')
        ax[2, 1].set_xlabel('Total radius [R⊙]')
        ax[2, 1].set_ylabel('Envelope temperature [K]')
        ax[2, 1].set_title('Temperature of the envelope')
        ax[2, 1].grid()

        ax[3, 0].plot(envelope.radius / R_sun, envelope.pressure,
                      color='green', label='dPdr')
        ax[3,0].plot(envelope.radius / R_sun,
                   (envelope.density / (mu * sc.m_u)) * sc.k * envelope.temperature,
                   label='P=rho*k*T/m_p',
                   color='blue',
                   linestyle='dotted')
        ax[3, 0].set_xlabel('Total radius [R⊙]')
        ax[3, 0].set_ylabel('Envelope pressure [Pa]')
        ax[3, 0].set_title('Pressure of the envelope')
        ax[3, 0].grid()
        ax[3, 0].legend()

    if full_return:
        return envelope, del_envelope, reduced_del_envelope, cor, re_cor
    elif ~full_return:
        return envelope, cor
