# Model design
import agentpy as ap
import numpy as np
import mpmath as mp

# Visualization
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation as animate
from IPython.display import HTML
import matplotlib.pyplot as plt
import pandas as pd

def normalize(v):
    """ Normalize a vector to length 1. """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def conensus_check(agent_opinions):
    """ Check whether agents have reached a consensus """
    consensus = False
    # Check population size
    population = len(agent_opinions)
    # Count number of agents for h1 and h2 respectively
    h1_count = np.count_nonzero(agent_opinions >= 0.9)
    h2_count = np.count_nonzero(agent_opinions <= 0.1)
    # Check if consensus reached
    if h1_count >= 0.9*population or h2_count >= 0.9*population:
        consensus = True
    return consensus

class Boid(ap.Agent):
    """ 
    An agent with a position and velocity in a continuous space,
    who follows Craig Reynolds' three rules of flocking behavior;
    plus a fourth rule to avoid the edges of the simulation space.
    Agents reach consensus 
    """

    def setup(self):
        """
        Initialise agents
        """
        # Initialise random velocity and opinion
        self.velocity = normalize(
            self.model.nprandom.random(self.p.ndim) - 0.5)
        self.opinion = mp.mpmathify(self.model.random.random())

    def setup_pos(self, space):
        """
        Set up agents position in space
        """
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def update_velocity(self):
        """
        Update velocity using Craig Reynolds' rules
        """
        # Get position and number of dimensions
        pos = self.pos
        ndim = self.p.ndim

        # Rule 1 - Cohesion
        nbs = self.neighbors(self, distance=self.p.outer_radius)
        nbs_len = len(nbs)
        nbs_pos_array = np.array(nbs.pos)
        nbs_vec_array = np.array(nbs.velocity)
        if nbs_len > 0:
            center = np.sum(nbs_pos_array, 0) / nbs_len
            v1 = (center - pos) * self.p.cohesion_strength
        else:
            v1 = np.zeros(ndim)

        # Rule 2 - Seperation
        v2 = np.zeros(ndim)
        for nb in self.neighbors(self, distance=self.p.inner_radius):
            v2 -= nb.pos - pos
        v2 *= self.p.seperation_strength

        # Rule 3 - Alignment
        if nbs_len > 0:
            average_v = np.sum(nbs_vec_array, 0) / nbs_len
            v3 = (average_v - self.velocity) * self.p.alignment_strength
        else:
            v3 = np.zeros(ndim)

        # Rule 4 - Borders
        v4 = np.zeros(ndim)
        d = self.p.border_distance
        s = self.p.border_strength
        for i in range(ndim):
            if pos[i] < d:
                v4[i] += s
            elif pos[i] > self.space.shape[i] - d:
                v4[i] -= s

        # Update velocity
        self.velocity += v1 + v2 + v3 + v4
        self.velocity = normalize(self.velocity)

    def update_position(self):
        """
        Updates the position of an agent with the updated velocity
        computed above
        """
        self.space.move_by(self, self.velocity)

    def pool_opinions(self):
        """
        Pool the opinions from nearby agents
        and update opinion
        """
        # Gather neighbours
        nbs = self.neighbors(self, distance=self.p.inner_radius)
        if len(nbs) > 0:
            # Generate random number between 0 and 1 (probability)
            prob = self.model.random.random()
            # Receive evidence if prob < epsilon
            if prob <= self.p.pooling_epsilon:
                nbs_ops_array = np.array(nbs.opinion)
                # Add back in own opinion
                pool_array = np.append(nbs_ops_array, self.opinion)
                h1 = (np.prod(pool_array))**self.p.w
                h2 = (np.prod(1-pool_array))**self.p.w
                # Update opinion using SProdOp
                self.opinion = h1/(h1+h2)

    def evidential_updating(self):
        """
        Update opinions from evidence at an epsilon chance 
        """
        # Generate random number between 0 and 1 (probability)
        prob = self.model.random.random()
        # Receive evidence if prob < epsilon
        if prob <= self.p.evidence_epsilon:
            # Upate using Chanelle's equation
            self.opinion = ((1-self.p.alpha)*self.opinion)/(self.opinion+self.p.alpha-2*self.p.alpha*self.opinion)


class BoidsModel(ap.Model):
    """
    A simulation to explore consensus forming in continuous space,
    movement based off the docs model from AgentPy.

    Animals' flocking behavior
    based on Craig Reynolds' Boids Model [1]
    and Conrad Parkers' Boids Pseudocode [2].
    Original code for movement and animation written by
    Joël Foramitti [3]

    [1] http://www.red3d.com/cwr/boids/
    [2] http://www.vergenet.net/~conrad/boids/pseudocode.html
    [3] https://agentpy.readthedocs.io/en/latest/agentpy_flocking.html 
    """

    def setup(self):
        """ 
        Initializes the agents and network of the model
        """
        # Initialise space and add agents to positions in space
        self.space = ap.Space(self, shape=[self.p.size]*self.p.ndim)
        self.agents = ap.AgentList(self, self.p.population, Boid)
        self.space.add_agents(self.agents, random=True)
        self.agents.setup_pos(self.space)

    def step(self):
        """ 
        Defines the models' events per simulation step
        """
        # Adjust direction
        self.agents.update_velocity()
        # Move into new direction
        self.agents.update_position()
        # Pool opinions with nearby agents
        self.agents.pool_opinions()
        # Evidential updating
        self.agents.evidential_updating()

    def update(self):
        
        # Create variable for agents opinions
        opinions = self.agents.opinion
        # Record agents opinions
        self.record("opinions", tuple(opinions))

        # Get agent's positions
        pos = self.space.positions.values()
        pos = np.array(tuple(pos)).T
        self.record("pos", pos)

        # Check if consensus reached
        consensus = conensus_check(opinions)
        if consensus:
            self.stop()
        return 

""" vvv Animation vvv """

def animate_plot_single(t, ax, sim_data):
    # Clear axis after each iteration
    if t > 0:
        for axis in ax:
            axis.clear()
    # Extract data
    pos = sim_data['pos'][t]
    opinions = np.asarray(sim_data['opinions'][t], dtype=float)
    # Set up main simulation scatter
    im = ax[0].scatter(*pos, s=10, c=opinions, cmap='cool', vmin=0, vmax=1)
    # Set axes limits and turn off numbers
    ax[0].set_xlim(0, 50)
    ax[0].set_ylim(0, 50)
    ax[0].set_axis_off()
    # Set up colourbar
    cb = plt.gcf().colorbar(im, cax=ax[1], orientation='vertical', fraction=0.05);
    avg_opinion = opinions.mean()
    cb.ax.hlines(avg_opinion,0,1,colors='k',linewidths=3)

def animate_plot(sim_data):
    # Set up figure
    fig = plt.figure(figsize=(8,8))
    simulation = fig.add_subplot(111)
    divider = make_axes_locatable(simulation)
    colourbar = divider.append_axes("right", size="5%", pad=0.05)
    ax = simulation, colourbar
    # Get the animation
    animation = animate(fig, animate_plot_single, frames=len(sim_data.index), fargs=(ax, sim_data))
    return HTML(animation.to_jshtml(fps=20))

""" vvv Run model vvv """

def run_sim(m, p):
    """ Run model and collect results, optional argument to produce animation """
    # Run model and collect results 
    model = m(p)
    results = model.run()
    return model, results