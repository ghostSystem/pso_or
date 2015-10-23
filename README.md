# pso_or
import numpy as np
import matplotlib.pylab as plt
from math import *
from mpl_toolkits.mplot3d import Axes3D


class Particle(object):
    """ Class representing probe particle
    """

    def __init__(self, x, v):
        """Constructor for Particle
        Keyword arguments:
        x -- double[d] coordinates of particles in d dimensions
        v -- double[d] velocity vector of particles in d dimensions
        """
        self.position = x
        self.velocity = v
        self.best_value = 0
        self.best_position = self.position
        self.value = 0

    def get_value(self, fun):
        """Method getting value of function fun in current position
        Keyword arguments:
        fun -- function defined as other method
        """
        value = fun(self.position)
        if value < self.best_value:   # minimisation option
            self.best_value = value
            self.best_position = self.position
        self.value = value

    def update_position(self, dt=1):
        """Method updating position of particle in current moment
        Keyword arguments:
        dt -- time step, default=1
        """
        self.position += np.array(self.velocity) * dt

    def update_velocity(self, global_best, coefficients):
        """Method updating velocity of particle in current moment
        Keyword arguments:
        global_best -- double[d] current best position of population, d-dimensions
        coefficients -- double[3] vector of three coefficients of simulation: 0-inertial 1-egoism 2-group terms
        """
        for i in range(len(global_best)):
            self.velocity[i] = self.velocity[i] * coefficients[0] + \
                coefficients[1] * np.random.random() * (self.best_position[i] - self.position[i]) + \
                coefficients[2] * np.random.random() * (global_best[i] - self.position[i])


def function(position):
    """Definition of function to minimise
            position -- double[2] position of particle in d=2 dimensions
        Returns:
            value of function
    """
    x = position[0]
    y = position[1]
    return 20 + exp(1) - 20*exp(-0.2*((x**2+y**2)/2)**0.5)-exp(1/2*(cos(2*pi*x)+cos(2*pi*y)))


#initialisation of particles
N = 10 # number of particles
dt = 0.1 # time step
particles = []
x_begin = y_begin = -8
x_end = y_end = 8
best_value = 0

for i in range(N):
    particles.append(Particle(2 * x_end * np.random.random(2) - x_end, [0, 0]))  # case of random position, no velocity!

c = [0.6, 0.5, 0.9]   # [inertial term, egoism term, obedience term]
number_of_steps = 10

#plot initialisation
x = np.linspace(x_begin, x_end, 100)
y = np.linspace(y_begin, y_end, 100)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
plt.ion()
plt.show()
Z = np.array([function((x, y)) for x, y in zip(np.ravel(X), np.ravel(Y))])  # calculating values of function
Z = Z.reshape(X.shape)

# main loop
for i in range(number_of_steps):
    values = []
    ax = fig.add_subplot(111, projection='3d')

    for particle in particles:  # calculation loop
        particle.get_value(function)
        values.append(particle.value)

    for particle in particles:  # update loop
        if min(values) < best_value:
            print('New minimum found: '+str(min(values)))
        best_value = min(values)

        particle.update_velocity(particles[values.index(min(values))].position, c)
        particle.update_position(dt)
        ax.scatter(particle.position[0], particle.position[1], particle.value, s=50, c='y')

    # plotting surface with transparency alpha = 0.2
    ax.plot_surface(X, Y, Z, alpha=0.2)
    plt.draw()
    plt.clf()
