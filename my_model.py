import random
import numpy as np

""" vvv Agent class for simulation and function to initialise agents vvv """

class Agent():
    """
    Class for agents involved in the simulation

    Initialised with a random opinion and a given weight
    """
    def __init__(self, w):
        self.opinion = random.uniform(0,1)
        self.weight = w
        pass

def init_agents(num_agents, w):
    """
    Initialise list of agents to work with
    """
    agents = []
    for i in range(num_agents):
        agents.append(Agent(w))
    return agents

""" vvv Functions for pooling agents vvv """

def rand_pop(list):
    """
    Pop random element from a list
    """
    # Get random index
    i = random.randrange(len(list))
    # Swap with the last element
    list[i], list[-1] = list[-1], list[i]    
    # Pop last element
    x = list.pop()
    return list, x

def extract_opinions(agent_pool):
    """
    Extract opinions of pooled agents for use with SProdOp
    """
    pool_opinions = []
    for agent in agent_pool:
        pool_opinions.append(agent.opinion)
    return pool_opinions

def SProdOp(pooled_opinions, w):
    """
    Performs SProdOp for some amount of opinions and a weight w
    """
    c = (np.prod(pooled_opinions)**w)/((np.prod(pooled_opinions)**w)+(np.prod(list(1-np.asarray(pooled_opinions)))**w))
    return c

def pool_agents(agents, k, w):
    """
    Picks out k agents at random and pools their probabilities using SProdOp
    """
    # Take out agents to pool probabilities
    agent_pool = []
    for i in range(k):
        agents, popped_agent = rand_pop(agents)
        agent_pool.append(popped_agent)
    # Extract opinions of agents
    pooled_opinions = extract_opinions(agent_pool)
    # Combine using SProdOp
    c = SProdOp(pooled_opinions, w)
    # Reassign agents with new pooled opinion and add back to main population
    for agent in agent_pool:
        agent.opinion = c
        agents.append(agent)
    return agents

""" vvv Functions for direct evidence given to agents vvv """

def update_op(agent, alpha):
    """
    Update opinion using Chanelle's equation
    """
    # Assign agent opinion to variable x to make equation shorter
    x = agent.opinion
    # Update opinion
    agent.opinion = (1-alpha)/(x+alpha-2*alpha*x)
    return agent

def choose_update(agent, epsilon, alpha):
    """
    Choose which agents to update probabilistically and update them
    """
    prob = random.uniform(0,1)
    if prob <= epsilon:
        agent = update_op(agent, alpha)
    return agent

def dir_evidence(agents, epsilon, alpha):
    """
    Loops through agents and updates their opinion with new evidence at an epsilon chance
    """
    for agent in agents:
        agent = choose_update(agent, epsilon, alpha)
    return agents

""" vvv Checking consensus function vvv """

def consensus_check(agents):
    """
    Check if consensus has been reached among agents
    returns 1 if consensus reached, 0 otherwise
    """
    num_agents = len(agents)
    h1_count = 0
    # Loop through agents and check their opinion
    for agent in agents:
        if agent.opinion < 0.9:
            h1_count += 1
        # If number of agents who haven't reach 90% belief in h1 reaches more than 10% consensus not reached
        if h1_count > num_agents//10:
            return 0
    return 1
        
""" vvv Main function vvv """

def run_sim(num_agents, k, alpha, epsilon, w):
    """
    Simulating the best-of-n problem

    Parameters
    ----------
    num_agents : int
        number of agents to simulate
    k : int
        pool-size for opinion pooling
    alpha : float
        reliability of evidence quantifier, must be between 0 and 0.5
    epsilon : float
        chance of agent recieving evidence at a given time step
    w : float
        weight associated with confidence in agents (fixed across all agents for SProdOp)
    """
    # Initialise agents and time counter
    agents = init_agents(num_agents, w)
    step = 0
    # Run through 10000 time steps for simulation
    while step < 10000:
        # Shuffle agents
        random.shuffle(agents)
        # Pick out k agents at random and pool their probabilities
        agents = pool_agents(agents, k, w)
        # Give evidence to each agent at a two percent chance
        agents = dir_evidence(agents, epsilon, alpha)
        # Check consensus levels
        success = consensus_check(agents)
        # Break out of loop if consensus reached otherwise increase step
        if success:
            break
        step += 1
    return step