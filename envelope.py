import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
import scipy.constants as sc
import core3

rho_core_sun = 1.62e5
rho_mean_sun = 1406.3134
R_sun = 6.9634*1e8
M_sun = 2*1e30
L_sun = 3.828e26

# Calculate core and take initial values
x_max = -1

rho_core = 1e6
T_core = 1e7
Y_e = 0.5
gaunt = 1
guillotine = 10 

cor, re_core = core3.solve(rho_core, T_core, Y_e)

rho_o = cor.density[-1]
m_o = cor.mass[-1]
P_o = cor.pressure[-1]
R_o = cor.radius[-1]
T_o = cor.temperature[-1]

rho_r =1e4
R_r = 1e4
P_r = P_o
T_r = T_o

q_o = rho_o/rho_r
M_o = m_o/((4/3)*sc.pi*R_r**3*rho_r)
p_o = P_o/P_r
x_o = R_o/R_r
t_o = T_o/T_r

if x_max == -1:
  x_max = 1e5/R_r
 

def luminosity(m, T):
  L = (T/(7e7))**(7/2)*(m/M_sun)*L_sun
  return L

def opacity(rho, T):
  kappa = 4.34e19*rho*T**(-7/2)
  return kappa

def envelope_equations(x, variables):
  q, M, t = variables
  kappa = opacity((q+q_o)*rho_r, T_r*(t+t_o))
  Cq = 4*sc.G*R_r**2*sc.m_p*sc.pi*rho_r/(3*T_r*sc.k)
  Ct = 3*L_r*kappa*l*rho_r/(64*R_r*T_r**4*sc.pi*sc.sigma)

  derivatives = [-Cq*(M + M_o)*(q + q_o)/((t + t_o)*(x + x_o)**2),
                 3*q*x**2,
                 (q + q_o)/((t + t_o)**3*(x + x_o)**2)]
  return derivatives

L_r = luminosity(m_o, T_o)
l = 1

env = solve_ivp(envelope_equations, [0,x_max], [0,0,0], method = 'RK45')

class envelope:
  mass = m_o + env.y[1]*(4/3)*sc.pi*R_r**3*rho_r
  mass = mass[~np.isnan(mass)]
  density = q_o + env.y[0]*rho_r
  density = density[0:len(mass)]
  temperature = t_o+env.y[2]*T_r
  temperature = temperature[0:len(mass)]
  radius = R_o + env.t*R_r
  radius = radius[0:len(mass)]
class del_envelope:
  mass = env.y[1]*(4/3)*sc.pi*R_r**3*rho_r
  mass = mass[~np.isnan(mass)]
  density = env.y[0]*rho_r
  density = density[0:len(mass)]
  temperature = env.y[2]*T_r
  temperature = temperature[0:len(mass)]
  radius = env.t*R_r
  radius = radius[0:len(mass)]
class reduced_del_envelope:
  mass = env.y[1]
  mass = mass[~np.isnan(mass)]
  density = env.y[0]
  density = density[0:len(mass)]
  temperature = env.y[2]
  temperature = temperature[0:len(mass)]
  radius = env.t
  radius = radius[0:len(mass)]
  
fig, ax = plt.subplots(3,2, dpi = 150, figsize=(45,15))

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

ax[1,1].plot(envelope.radius, envelope.density, color = 'orange')
ax[1,1].set_xlabel('Total radius [m]')
ax[1,1].set_ylabel('Envelope density [kg/m^3]')
ax[1,1].set_title('Density of the envelope')
ax[1,1].grid()

ax[2,0].plot(envelope.radius, envelope.mass, color = 'orange')
ax[2,0].set_xlabel('Total radius [m]')
ax[2,0].set_ylabel('Envelope mass [kg]')
ax[2,0].set_title('Mass of the envelope')
ax[2,0].grid()

ax[2,1].plot(envelope.radius, envelope.temperature, color = 'orange')
ax[2,1].set_xlabel('Total radius [m]')
ax[2,1].set_ylabel('Envelope temperature [K]')
ax[2,1].set_title('Temperature of the envelope')
ax[2,1].grid()
