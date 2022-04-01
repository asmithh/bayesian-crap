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
)
from utilities import markov_update_log, make_configuration_model_graph


def run_agent_simulation(N_AGENTS, params_dict):
    """
    Given a number of agents & parameters for constructing the simulation,
    run a 100-round simulation of belief updating w/ Bayesian agents.
    
    Returns info on each agent's parameters, sharing for each agent each round, and centrality info for each agent.
    """
    agents = []

    for i in range(N_AGENTS):
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

    # G, agents = make_er_graph(0.05, N_AGENTS, agents)
    G, agents = make_configuration_model_graph(N_AGENTS, 2.5, agents, params_dict)
    centrality = sorted(
        [(k, v) for k, v in nx.closeness_centrality(G).items()], key=lambda b: b[0]
    )

    centrality = np.array([c[1] for c in centrality]).reshape(-1, 1)
    agent_records = {a.agent_id: {} for a in agents}
    shares = {a.agent_id: {} for a in agents}

    # from multiprocessing import Pool
    # pool = Pool(8)

    for time_step in range(250):
        for agent in agents:
            agent_records[agent.agent_id][time_step] = {
                "neighbor_trust": agent.neighbors,
                "misinfo_belief": agent.misinfo_belief,
                "share_propensity": agent.share_propensity,
            }

        neighbor_beliefs = [
            [(i, agents[i].misinfo_belief) for i in agent.neighbors.keys()]
            for agent in agents
        ]
        neighbor_forcefulness = [
            [agents[i].forcefulness for i in agent.neighbors.keys()] for agent in agents
        ]
        agent_info_dicts = [
            make_agent_info_dict(a, b, f, params_dict)
            for a, b, f in zip(agents, neighbor_beliefs, neighbor_forcefulness)
        ]
        res = map(update_agent_info, agent_info_dicts)
        for r, agent in zip(res, agents):
            agent.neighbors = r["neighbor_trust"]
            agent.misinfo_belief = r["misinfo_belief"]
            agent.share_propensity = r["share_propensity"]
            shares[agent.agent_id][time_step] = r["shares"]

    return agents, shares, centrality


def p_x_y(agents, shares, centrality, alpha):
    loss = 0.0
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
    
    
    shared_by_top_one_percent_model = np.sum(
        sorted(shared)[-int(0.01 * len(shared)) :]
    ) / (1 + np.sum(shared))
    shared_by_top_one_percent_real = 0.8
    loss += np.abs(shared_by_top_one_percent_model - shared_by_top_one_percent_real) ** alpha

    
    n_shared_per_capita_model = np.sum(shared) / len(agents)
    n_shared_per_capita_real = 1.0
    loss += np.abs(n_shared_per_capita_model - n_shared_per_capita_real) ** alpha
    
    return loss / alpha
    
    
def G_func(my_ensemble_P, x):
    # make fancier later? right now assumes constant prior (the horror)
    constant_proba = np.log(0.1) * 10 + np.log(0.01) * 2
    candidates = [np.exp(constant_proba) for tup in my_ensemble_P if tup[1] <= x]
    return np.sum(candidates)
            

N_AGENTS = 200
ALPHA = 2.5
EPSILON_INIT = 0.5
rnd_info = []

ensemble_P = []
ensemble_E = []

while len(ensemble_E) < 200:
    if len(ensemble_E) % 5 == 0 and len(ensemble_E) != 0:
        print(len(ensemble_E))
    params_dict = generate_params_dict()
    agents, shares, centrality = run_agent_simulation(N_AGENTS, params_dict)
    tup = (params_dict, p_x_y(agents, shares, centrality, ALPHA))
    proba_p = np.exp(-1.0 * tup[1]/ EPSILON_INIT)
    draw = np.random.uniform()
    if draw < proba_p:
        ensemble_E.append(tup)
    ensemble_P.append(tup)
    
G_result = [G_func(ensemble_P, tup[1]) for tup in ensemble_P]
ensemble_E = [(tup[0], G_func(ensemble_P, tup[1])) for tup in ensemble_E]
U = np.mean(G_result)

EPSILON = EPSILON_INIT
t = 1
swap_rate = 0
tries = 0
while True:
    chosen_one = random.choice([i for i in range(len(ensemble_E))])   
    particle, u = ensemble_E[chosen_one
                            ]
    proposal = generate_params_dict()
    agents, shares, centrality = run_agent_simulation(N_AGENTS, params_dict)
    proba_star = p_x_y(agents, shares, centrality, ALPHA)
    u_star = G_func(ensemble_P, proba_star)
    
    proba_swap = min(1.0, np.exp(-1.0 * (u_star - u)) / np.exp(EPSILON))
    tries += 1
    if np.random.uniform() < proba_swap:
        swap_rate += 1
        ensemble_E[chosen_one] = (proposal, u_star)

    EPSILON = t ** (0.1)
    if t % 10 == 0:
        print(swap_rate / tries)
    if t % 100 == 0:
        pickle.dump(ensemble_E, open('ensemble_E_{}.pkl'.format(str(t)), 'wb'))
    t += 1
    if (swap_rate / tries) < 0.05 and tries > 20:
        break
