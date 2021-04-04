'''
CALCULATES THE COOLNG TRACK OF A WHITE DWARF
'''

import core
import envelope
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.constants as sc

rho_core_sun = 1.62e5
rho_mean_sun = 1406.3134
R_sun = 6.9634 * 1e8
M_sun = 2 * 1e30
L_sun = 3.828e26
halfsolardens = 1.9677e9
m_u = 1.6605390666e-27
million_years = 1e6 * 365 * 24 * 3600


def calculate(
        M,
        T_o,
        R,
        Y_e_core,
        mu_core,
        X,
        Y,
        Z,
        core_class,
        C,
        solver='RK23',
        r_tol=1e-3,
        a_tol=1e-6,
        graphs=False,
        t_max=100,
        alpha=0,
        beta=0,
        storing_times=None,
        crys=True):
    '''
    AUXILIARY FOR FULL_CALCULATE
    '''
    cor = core_class
    kappa_o = 4.34e23 * Z * (1 + X)
    mu_envelope = 2 / (1 + 3 * X + 0.5 * Y)
    mu_e_core = 1 / Y_e_core
    mu_ion_core = ((1 / mu_core) - (1 / mu_e_core))**(-1)
    Cv = (3 / 2) * sc.k
    Cv_crystal = 3 * sc.k
    mass = np.flip(cor.mass)
    if alpha == 0:
        def equations(t, T):

            for i, rho in enumerate(np.flip(cor.density)):
                gamma = (1 / (4 * sc.pi * sc.epsilon_0)) * (( mu_ion_core / 2 * sc.e)
                                                            ** 2 / (sc.k * T)) * (4 * sc.pi * rho / (3 * mu_ion_core * m_u))**(1 / 3)

                if gamma >= 171 and crys:
                    mass_energy = (
                        (mass[0] - mass[i]) * Cv + (mass[i]) * Cv_crystal) / (mu_ion_core * m_u)
                    break
                elif i + 1 == len(cor.density):
                    mass_energy = M * Cv / (mu_ion_core * m_u)

            rho_c = 6e-6 * mu_e_core * T**(3 / 2)
            L = (32 / (3 * 8.5)) * sc.sigma * (4 * sc.pi * sc.G * M / \
                 kappa_o) * mu_envelope * m_u / (sc.k) * T**(6.5) / (rho_c**2)

            dTdt = -1000 * million_years * L / mass_energy
            return dTdt
    else:
        def equations(t, T):
            for i, rho in enumerate(np.flip(cor.density)):
                gamma = (1 / (4 * sc.pi * sc.epsilon_0)) * (( mu_ion_core / 2 * sc.e)
                                                            ** 2 / (sc.k * T)) * (4 * sc.pi * rho / (3 * mu_ion_core * m_u))**(1 / 3)

                if gamma >= 171 and crys:
                    mass_energy = (
                        (mass[0] - mass[i]) * Cv + (mass[i]) * Cv_crystal) / (mu_ion_core * m_u)
                    break
                elif i + 1 == len(cor.density):
                    mass_energy = M * Cv / (mu_ion_core * m_u)

            rho_c = 6e-6 * mu_e_core * T**(3 / 2)
            L = (32 / (3 * 8.5)) * sc.sigma * (4 * sc.pi * sc.G * M / \
                 kappa_o) * mu_envelope * m_u / (sc.k) * T**(6.5) / (rho_c**2)

            dTdt = -1000 * million_years * (L + alpha * T**beta) / mass_energy
            return dTdt

    cool = solve_ivp(equations,
                     [0,
                      t_max],
                     [T_o],
                     method=solver,
                     rtol=r_tol,
                     atol=a_tol,
                     t_eval=storing_times)

    class evolution:
        time = cool.t * million_years * 1000
        core_temperature = cool.y[0]
        luminosity = (32 / (3 * 8.5)) * sc.sigma * (4 * sc.pi * sc.G * M / kappa_o) * \
            mu_envelope * m_u / (sc.k) * core_temperature**(3.5) / ((6e-6 * mu_e_core)**2)
        surface_temperature = (
            luminosity / (4 * sc.pi * R**2 * sc.sigma))**(1 / 4)

    if graphs:
        fig, ax = plt.subplots(2, 2, dpi=200, figsize=(20, 20))
        ax[0, 0].plot(evolution.time / (1000 * million_years),
                      evolution.core_temperature)
        ax[0, 0].set_xlabel('Billion years')
        ax[0, 0].set_ylabel('Core temperature [K]')
        ax[0, 0].grid()

        ax[0, 1].plot(evolution.time / (1000 * million_years),
                      evolution.surface_temperature)
        ax[0, 1].set_xlabel('Billion years')
        ax[0, 1].set_ylabel('Surface temperature [K]')
        ax[0, 1].grid()

        ax[1, 0].plot(evolution.time / (1000 * million_years),
                      evolution.luminosity / L_sun)
        ax[1, 0].set_xlabel('Billion years')
        ax[1, 0].set_ylabel('Solar luminosities')
        ax[1, 0].grid()

    return evolution


def full_calculate(
        rho_core,
        T_core,
        Y_e_core,
        C,
        X,
        Y,
        Z,
        solver='RK23',
        r_tol=1e-3,
        a_tol=1e-6,
        graphs=False,
        t_max=14,
        full_output=False,
        storing_times=None,
        alpha=0,
        beta=0,
        crys=True):
    '''
    Description
    -----------
    CALCULATES THE COOLING TRACK OF A WHITE DWARF RELYING ON THE CORE
    AND ENVELOPE MODULES TO FIRST SET THE STRUCTURE OF THE STAR.

    C/O CORE.

    INCLUDES THE CRYSTALLIZATION OF THE CORE WITH THE CRYSTALLIZATION
    FRONT ADVANCING TOWARDS THE ENVELOPE.

    DOES NOT INCLUDE DEBYE COOLING.

    Parameters
    ----------

    rho_core: float
        Density of the core at r=0 in kg/m^3.
    T_core: float
        Temperature of the isothermal core in K.
    Y_e_core: float
        Free electrons per nucleon of the core.
        For C/O core Y_e_core = 0.5
    C: float
        Mass fraction of carbon in the C/O core.
    X: float
        Hydrogen mass fraction FOR THE ENVELOPE.
    Y: float
        Helium mass fraction FOR THE ENVELOPE.
    Z: float
        Metals mass fraction FOR THE ENVELOPE.
    solver: string, optional
        Which method to use for solve_ivp.
        Default is 'RK23'.
    r_tol float, optional
        Maximum relative error
        For detailed information refer to documentation of
        'solve_ivp' from scipy.
        Default is 1e-3.
    a_tol: float, optional
        Maximum absolute error
        For detailed information refer to documentation of
        'solve_ivp' from scipy.
        Default is 1e-6.
    graphs: boolean, optional
        Whether to print graphs or not. False by default.
    t_max: float, optional
        Time in billion years up to which calculate cooling.
        Default is 14 billion years.
    full_output: boolean, optional
        Whether to output the core and envelope classes.
        Default is False
    storing_times: array
        One dimensional array containing the times in seconds
        at which to store results.
        Default is None, leaving the integrator to decide when
        to store results.
    alpha, beta = float, optional
        For the purpose of testing the addition of an extra
        cooling mechanism of the for alpha*T^beta.
        Default is 0.
    crys: boolean, optional
        For test purposes, whether to include crystallization
        in the cooling or not.
        Default is True.

    Returns
    -------
    evolution: class
        Contain the values for different properties of the star:

        evolution.time                | Time in seconds
        evolution.luminosity          | Evolution of the star luminosity in W
        evolution.core_temperature    | Evolution of the core temperature
        evolution.surface_temperature | Evolution of the surface temperature

    Additionally if full_return it will also return the core and envelope classes from the envelope module as

    evolution, envelope, core

    '''

    mu_core = ((1 / 48) * C + 9 / 16)**(-1)

    env, cor = envelope.solve(rho_core, T_core, Y_e_core, X, Y, Z, solver=solver, r_tol_core=r_tol,
                              r_tol_envelope=r_tol, a_tol_core=a_tol, a_tol_envelope=a_tol, message=False)

    evolution = calculate(cor.mass[-1],
                          cor.temperature[-1],
                          env.radius[-1],
                          Y_e_core,
                          mu_core,
                          X,
                          Y,
                          Z,
                          cor,
                          C,
                          graphs=graphs,
                          solver=solver,
                          t_max=t_max,
                          storing_times=storing_times,
                          r_tol=r_tol,
                          a_tol=a_tol,
                          alpha=alpha,
                          beta=beta,
                          crys=crys)

    if full_output:
        return evolution, env, cor
    elif ~full_output:
        return evolution
