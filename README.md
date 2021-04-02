# A COMPUTATIONAL MODEL OF WHITE DWARF COOLING

This is a simple python model to calculate both the structure and cooling of white dwarfs.

## Description

The project is composed of three modules: core, envelope and cooling.

- Core solves the core structure of the white dwarf.
- Envelope solves the core plus the envelope of the white dwarf.
- Cooling solves the whole structure plus the cooling of the white dwarf.

For instructions on how each module work refer to instructions

## Getting Started

### Dependencies

* The code is written in python 3
* numpy
* scipy
* matplotlib
* Modules depend on eachother. That is, cooling needs envelope and core; envelope needs core; core needs the dependencies listed above.

### Installing

* Just download the three modules and import them into python using:
- import core
- import envelope
- import cooling

## Instructions

### Core module
#### core.solve
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
    graphs: boolean, optional
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
    
    core, reduced_core
    
     core: class
        Contains values for the core in increasing radius according to
        
        core.mass        | Mass in kg, array
        core.density     | Density in kg/m^3, array
        core.temperature | Temperature in K, array
        core.pressure    | Pressure in Pa, array
        core.radius      | Radius in m, array
        
    reduced_core: class
        Same as core but values are reduced

### Envelope module
#### envelope.solve

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
#### envelope.lum

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
### Cooling modules
#### cooling.full_calculate
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
## Authors
Julio Hernandez Camero

julhcam@gmail.com

## Version History
* 0.1
    * Initial Release

## License

This project is licensed under the [MIT] License - see the LICENSE.md file for details
