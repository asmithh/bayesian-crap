import pickle
import random
from sklearn.linear_model import LinearRegression

import networkx as nx
import numpy as np

from agents import misinfoAgent
from misinfo_functions import (
    generate_params_dict,
    step_params_dict,
    calc_energy,
    acceptance_proba,
    make_agent_info_dict,
    update_agent_info,
    markov_update_params_dict,
)
from utilities import markov_update_log, make_powerlaw_cluster_graph, make_er_graph, make_configuration_model_graph

import sys

######
# explains the arguments; right now you do python3 misinfo_abc.py [graph_type] [pickle_title]
# graph_type is "er", "pwrlaw", or "config"
# pickle_title is just a way to title your parameter pickles so that you can keep track of runs.
# hopefully pickle_title changes between runs.
GRAPH_TYPE = sys.argv[1].strip()
PICKLE_TITLE = sys.argv[2].strip()
######

# the skeleton code from this is probably fine to copypasta to make your own function that runs agent simulations.
# you'll want to change how the agents are initialized and then change the updated params/the update function in the for loop.
def run_agent_simulation(N_AGENTS, params_dict):
    """
    Given a number of agents & parameters for constructing the simulation,
    run a 100-round simulation of belief updating w/ Bayesian agents.
    
    Returns info on each agent's parameters, sharing for each agent each round, and centrality info for each agent.
    """
    agents = []

    for i in range(N_AGENTS):
        # agents get initialized here!
        agent = misinfoAgent(
            agent_id=i,
            neighbors={},
            forcefulness=np.log(
                np.random.beta(params_dict["B1_START_FO"], params_dict["B2_START_FO"])
            ),
            share_propensity=np.log(
                np.random.beta(params_dict["B1_START_SP"], params_dict["B2_START_SP"])
            ),
            misinfo_belief=np.log(
                np.random.beta(params_dict["B1_START_MB"], params_dict["B2_START_MB"])
            ),
            trust_stability=np.log(
                np.random.beta(params_dict["B1_START_TS"], params_dict["B2_START_TS"])
            ),
        )
        agents.append(agent)
    if GRAPH_TYPE == 'er':
        G, agents = make_er_graph(0.05, N_AGENTS, agents, params_dict)
    elif GRAPH_TYPE == 'config':
        G, agents = make_configuration_model_graph(N_AGENTS, 2.5, agents, params_dict)
    elif GRAPH_TYPE == 'pwrlaw':
        G, agents = make_powerlaw_cluster_graph(N_AGENTS, agents, 0.05)
    # you may not need to calculate centrality in your graph; feel free to get rid of this if that's the case.
    centrality = sorted(
        [(k, v) for k, v in nx.closeness_centrality(G).items()], key=lambda b: b[0]
    )

    centrality = np.array([c[1] for c in centrality]).reshape(-1, 1)
    agent_records = {a.agent_id: {} for a in agents}
    shares = {a.agent_id: {} for a in agents} 
    # your agents might do something besides sharing; feel free to rename / restructure accordingly.

    # from multiprocessing import Pool
    # pool = Pool(8)

    for time_step in range(250):
        for agent in agents:
            # we're just keeping track of drift in individual agent parameters over time.
            agent_records[agent.agent_id][time_step] = {
                "neighbor_trust": agent.neighbors,
                "misinfo_belief": agent.misinfo_belief,
                "share_propensity": agent.share_propensity,
            }
        # grabbing the info we'll need to update agent info; this will change depending on your update function.
        neighbor_beliefs = [
            [(i, agents[i].misinfo_belief) for i in agent.neighbors.keys()]
            for agent in agents
        ]
        neighbor_forcefulness = [
            [agents[i].forcefulness for i in agent.neighbors.keys()] for agent in agents
        ]
        # you'll want to package up all the relevant info into one dict per agent that has everything you need to update.
        agent_info_dicts = [
            make_agent_info_dict(a, b, f, params_dict)
            for a, b, f in zip(agents, neighbor_beliefs, neighbor_forcefulness)
        ]
        res = map(update_agent_info, agent_info_dicts)
        # you'll want to make an update_agent_info function that works for your agents; this might be the most complex part of the process thus far.
        for r, agent in zip(res, agents):
            # update your agents' info according to the results of update_agent_info.
            agent.neighbors = r["neighbor_trust"]
            agent.misinfo_belief = r["misinfo_belief"]
            agent.share_propensity = r["share_propensity"]
            shares[agent.agent_id][time_step] = r["shares"]

    return agents, shares, centrality


def p_x_y(agents, shares, centrality, alpha):
    # this is kind of a loss function that tells us how far from reality our simulation result was.
    # i'll label the components etc, but the basic idea is the following:
    # we have a vector of parameters [p1, p2, p3....pk] that are characteristics of the simulation result.
    # mine are the fit between centrality & sharing, the fraction of misinfo shared by the top 1% of sharers,
    # and the amount of misinfo shared per capita.
    # we just do a simple |p1 - reality| ** alpha + |p2 - reality| ** alpha.... + |pk - reality| ** alpha
    # and then divide the result by alpha.
    # this gives us a fairly reasonable loss; feel free to tweak this if it doesn't work for you though!
    loss = 0.0

    # here we're correlating centrality in the network to misinfo sharing; 
    # we expect there to be a positive correlation that is medium strong.
    shared = [np.sum([v for v in shares[a.agent_id].values()]) for a in agents]
    shared_by_id = [
        (a.agent_id, np.sum([v for v in shares[a.agent_id].values()])) for a in agents
    ]
    shared_by_id = sorted(shared_by_id, key=lambda b: b[0])
    shared_by_id = [s[1] for s in shared_by_id]
    reg = LinearRegression().fit(centrality, shared)
    centrality_to_n_shared_model = reg.coef_[0]
    centrality_to_n_shared_real = 0.5
    loss += np.abs(centrality_to_n_shared_model - centrality_to_n_shared_real) ** alpha
    
    # we compare the percent of misinfo shared by the top 1% of sharers to the result from the lazer lab.
    shared_by_top_one_percent_model = np.sum(
        sorted(shared)[-int(0.01 * len(shared)) :]
    ) / (1 + np.sum(shared))
    shared_by_top_one_percent_real = 0.8
    loss += np.abs(shared_by_top_one_percent_model - shared_by_top_one_percent_real) ** alpha

    # just getting a ballpark estimate of how much misinfo is shared per user on average.
    n_shared_per_capita_model = np.sum(shared) / len(agents)
    n_shared_per_capita_real = 1.0
    loss += np.abs(n_shared_per_capita_model - n_shared_per_capita_real) ** alpha
    
    return loss / alpha
    
    
def G_func(my_ensemble_P, x):
    # this is a smoothed / normalized function that tells us what percentile our score X is
    # compared to the "background" ensemble P (a bunch of parameter sets drawn from the uniform prior)
    # higher G_func = better compared to the general population. 
    # 1 means it's the very very best
    # 0 means it's worse than everything we've ever seen.
    # 0.5 means it's middling.
    constant_proba = np.log(0.1) * 10 + np.log(0.01) * 2
    normalizer = len(my_ensemble_P) * np.exp(constant_proba)
    candidates = [np.exp(constant_proba) for tup in my_ensemble_P if tup[1] <= x]
    return np.sum(candidates) / normalizer

##################################################
# if you're going to mess with something, mess with these.
N_AGENTS = 100 # number of agents in the simulation. more is better but will take longer.
ALPHA = 2.8 # weight/exponent parameter for loss function. larger = bad fits penalized more
EPSILON_INIT = 0.3 # this is a decay parameter. it makes it less likely for us to take "risky" moves as time goes on;
# EPSILON will change exponentially with time in order to make this happen.
# i think that larger EPSILON_INIT loosely corresponds to a more selective initial ENSEMBLE_E.
K = np.ones((len(params_dict), len(params_dict))) # making the covariance matrix for our multivariate normal proposal distribution
K *= 0.05 # this is the background covariance for all parameters
for i in range(len(params_dict)):
    K[i, i] = 0.5 # the same variable is correlated with itself. no idea if this makes a difference.
    if i % 2 == 0 and i < 10:
        K[i, i + 1] = 0.25 # beta parameters are somewhat correlated with each other if they are for the same dist
        K[i + 1, i] = 0.25 # this makes the matrix symmetric.
        # feel free to design your covariance matrix however you want!

BETA = 0.98 # weighting parameter for updating the covariance matrix. should be close to 1
LITTLE_S = 0.02 # smoothing parameter for updating the covariance matrix K. should be small.
U_CONST = 1.0 # weights the O(U^2) term for the epsilon energy term.
GAMMA_V_RATIO = 0.2 # parameter that needs tuning; this is an approximation and ideally it updates as time goes on.
# keeping it constant isn't ideal but should work ok.
##################################################

rnd_info = []

ensemble_P = [] # this is our distribution that matches the uniform prior. it's kind of the "team everybody" team
ensemble_E = [] # this is our distribution of "good" parameter sets. it's like the olympic team?

while len(ensemble_E) < 250:
    # we keep trying to add things to ensemble_E until it's the length we want.
    if len(ensemble_E) % 5 == 0 and len(ensemble_E) != 0:
        print(len(ensemble_E)) # just keeping track of how many items we have in ensemble_E
    params_dict = generate_params_dict() # draw params from uniform prior
    agents, shares, centrality = run_agent_simulation(N_AGENTS, params_dict)
    tup = (params_dict, p_x_y(agents, shares, centrality, ALPHA)) # we run a simulation using these params and see how good it is
    proba_p = np.exp(-1.0 * tup[1]/ EPSILON_INIT) # we decide whether it's worthy of being in ensemble E
    draw = np.random.uniform()
    if draw < proba_p:
        ensemble_E.append(tup)
    ensemble_P.append(tup) # no matter what, we keep this param dict for ensemble E
    
G_result = [G_func(ensemble_P, tup[1]) for tup in ensemble_P] # smoothing probabilities in ensemble P
ensemble_E = [(tup[0], G_func(ensemble_P, tup[1])) for tup in ensemble_E] # same for ensemble E
U = np.mean(G_result)
print('going')
EPSILON = EPSILON_INIT # EPSILON is our tuning parameter. it controls how fast we stop making riskier decisions.
draws = []
t = 1
swap_rate = 0 # swap_rate is how often we swap out an item in ensemble_E for a proposed item.
# we usually use this as a stopping criterion; when swapping gets rarer, it's time to stop the algorithm.
tries = 0
while True:
    chosen_one = random.choice([i for i in range(len(ensemble_E))]) # choose a random item in ensemble E
    particle, u = ensemble_E[chosen_one] 
    proposal, new_vec, reject = markov_update_params_dict(particle, K) # do a markov update from this item (like MH ish)
    # reject is our auto-rejection that we do if any of the params is in an inappropriate range (e.g. beta param less than 0)
    agents, shares, centrality = run_agent_simulation(N_AGENTS, params_dict) 
    proba_star = p_x_y(agents, shares, centrality, ALPHA) # see how good it is
    u_star = G_func(ensemble_P, proba_star) # get smoothed score
    
    proba_swap = min(1.0, np.exp((-1.0 * (u_star - u)) / EPSILON)) # probability of keeping it
    tries += 1
    if np.random.uniform() < proba_swap and not(reject):
        # we keep it
        swap_rate += 1
        ensemble_E[chosen_one] = (proposal, u_star)

    draws.append(new_vec) # draws lets us keep track of the vectors we're drawing so we can adjust covariance accordingly.
    if t % 100 == 0:
        cov = np.cov(np.array(draws).transpose()) # adjusting covariance matrix K
        draws = []
        U = np.mean([G_func(ensemble_P, tup[1]) for tup in ensemble_E]) # how good our ensemble E is on average
        K = BETA * cov + LITTLE_S * np.trace(cov) * np.ones((len(params_dict), len(params_dict))) # adjusting K
        # note that this is where BETA and LITTLE_S come in.
        EPSILON = U ** (4/3.0) * (GAMMA_V_RATIO ** (1/3.0)) + U_CONST * U * U # adjusting EPSILON (less risky as time goes on)
        print(swap_rate / tries)
    if t % 100 == 0 or t % 5 == 0 and t < 75:
        # dumping ensemble E to disk; feel free to choose conditions for when this happens.
        # note that the filepath contains a unique keyword you specify in the command line
        # so that you can find all ensemble E files from a particular run.
        pickle.dump(ensemble_E, open('ensemble_E_{}_{}.pkl'.format(PICKLE_TITLE, str(t)), 'wb'))
    t += 1
    if (swap_rate / tries) < 0.001 and tries > 20:
        # stopping criterion; 0.001 takes a while to get to. feel free to increase.
        break
