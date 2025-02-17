# ==== Description ======== #
""""
Python code for wildfire model using the Euler method (standalone).

Author: Sebastian G. Mutz (sebastian@mutz.science)
License: MIT
"""
# ==== Import Packages ==== #

import math
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ==== Declarations ==== #

# ---- boundary conditions for your model
rho_wood0  = 100. # starting wood density (kg / m^2); average taking empty spaces into account
u          = 0.05 # u-wind component (m / s)
v          = 0.2  # v-wind component (m / s)
T_a        = 303. # atmospheric/ambient terature (K); we assume a warm summer day (~30°C)

# ---- constants
T_ig       = 570.       # ignition terature (K); we assume dry wood (~300°C)
T_burn     = 1400.      # burn terature (K); typically between 1300-1400K for wood

r_diff     = 5.         # diffusion coefficient (m^2 / s)
C_loss     = 120.       # heat loss constant (s)
C_burnTime = 0.5*3600.  # 0.5h in seconds (s)
heating    = (T_burn-T_a)/(C_loss*100.)*C_burnTime # K / (kg / m^2)

# ---- define your geographical domain
length = 1000.          # domain length (m)
size   = 200            # number of grid points
dx     = length/size    # step size

# --- experiment
h = 0.25*(dx**2/r_diff)  # time step size (s);
end_time = 60.*5.        # simulation time (s); 5 minutes
pos_i = -40.,220         # how/where to start the fire

# --- other important variables used later in the code
# rho_wood1 - old wood density
# rho_wood2 - new wood density
# T_f1  - old forest Terature
# T_f2  - new forest Terature

# ==== Instructions ==== #

# This is our function to start our forest fire.
def wildfire_start():

    # Set initial temperature to ambient temperature everywhere
    T_f1 = T_a*numpy.ones([size, size])  # K
    rho_wood1 = numpy.zeros([size, size]) # kg / m^2

    for j in range(0, size):
        for i in range(0, size):

            # convert grid points to real (physical) values
            x = i*dx-0.5*length+0.5*dx
            y = j*dx-0.5*length+0.5*dx

            # Starting the fire (don't do this in real life!): Here, an exponential function creates a
            # Gaussian-shaped temperature field centred at pos_i, which decreases as you move away from the center.
            # Adding T_a ensures that temperature far away from the centre is just the ambient temperature
            T_f1[j,i] = (T_burn-T_a)*math.exp(-((x+pos_i[0])**2+(y+pos_i[1])**2)/(2.*pos_i[0]**2))+T_a

            # Here, we initialise our wood density with the pre-defined value
            rho_wood1[j,i] = rho_wood0

    return T_f1, rho_wood1

# This is our function to simulate our wildfire. It returns updated forest
# temperatures and wood densities, which we later plot with wildfire_plot()
def wildfire_evolve(T_f1, rho_wood1):

    T_f2 = numpy.copy(T_f1)           # K
    rho_wood2 = numpy.copy(rho_wood1) # kg / m2
    num_steps = int(end_time/h)
    for step in range(num_steps):
        for j in range(1,size-1):
            for i in range(1,size-1):

                # Pass temeprature at current coordinates to T
                T = T_f1[j,i]

                # If below ignition temperature, set amount of burned wood and combustion to 0
                if T < T_ig:
                    wood_burn = 0
                    combustion = 0
                # If above ignition temperature, update amount of burned wood and combustion accordingly
                else:
                    # amount of wood burning per time
                    burn_rate = rho_wood1[j, i]/C_burnTime
                    # wood mass decreases as it burns (h = timestep)
                    wood_burn = -h*burn_rate
                    # The energy released by burning wood, contributing to temperature increase
                    combustion = h*heating*burn_rate

                # Below, we compute change in temperature and wood density due to heat diffusion, heat loss, wind, and combustion

                # This term models the spread of heat due to diffusion, following the discrete form of the heat equation.
                # The term inside parentheses sums up temperatures of neighboring grid points and subtracts 4T,
                # ensuring conservation of heat energy. The coefficient r_diff/dx**2 controls the rate of diffusion.
                heat_diffusion = h*r_diff/dx**2*(T_f1[j,i-1]+T_f1[j,i+1]+T_f1[j-1,i]+T_f1[j+1,i]-4*T)

                # Here, we compute the cooling based on heat loss to the surroundings.
                # The constant C_loss controls the rate at which heat is lost.
                heat_loss = h*(T_a-T)/C_loss

                # Here, we calculate heat advection due to wind in the x- and y-directions. The wind
                # velocity components u and v move the temperature field along their respective directions.
                wind_x = -h*u/dx*0.5*(T_f1[j,i+1]-T_f1[j,i-1])
                wind_y = -h*v/dx*0.5*(T_f1[j+1,i]-T_f1[j-1,i])

                # Update wood density based on what was burned.
                rho_wood2[j,i] = rho_wood1[j,i]+wood_burn

                # Update the forest temperature based on a combination of all the temperature modifying
                # factors calculated above.
                T_f2[j,i] = T+combustion+heat_diffusion+heat_loss+wind_x+wind_y

        # Pass updated forest temperature and wood density
        T_f1,T_f2 = T_f2,T_f1
        rho_wood1,rho_wood2 = rho_wood2,rho_wood1

    return T_f1, rho_wood1

# This function plots our modelled results: A map of temperature and a map of wood density
def wildfire_plot():
    # Set dimensions
    dimensions = [-0.5*length, 0.5*length, -0.5*length, 0.5*length]

    # temperature plot
    axes = plt.subplot(221)
    plt.imshow(T_f1, cmap=cm.hot, origin='lower', vmin=200, vmax=1300)
    plt.colorbar()
    axes.set_title('temperature ($K$)')
    axes.set_xlabel('x (m)')
    axes.set_ylabel('y (m)')

    # wood density plot
    axes = plt.subplot(222)
    plt.imshow(rho_wood1, cmap=cm.summer_r, origin='lower', vmin=50, vmax=100)
    plt.colorbar()
    axes.set_title('wood density ($kg/m^2$)')
    axes.set_xlabel('x (m)')
    axes.set_ylabel('y (m)')
    plt.show()

# Here, we start the fire. The function returns an updated forest temperaturee and wood density.
T_f1, rho_wood1 = wildfire_start()

# Here, we activate the function that lets our fire evolve. We pass the updated values to the function,
# run the model for the previously defined time, and finally get back the values for temperature and
# wood density that represents a snapshot of these values at the end of our simulation.
T_f1, rho_wood1 = wildfire_evolve(T_f1, rho_wood1)

# Here, we call the plot function to visualise our results.
wildfire_plot()

