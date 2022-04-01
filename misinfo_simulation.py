import pickle
import random

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
        [(k, v) for k, v in nx.eigenvector_centrality(G).items()], key=lambda b: b[0]
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


N_AGENTS = 250
SA_STEPS = 250
rnd_info = []

for rnd in range(1000):
    print(rnd)
    energy = []
    params_dict_old = generate_params_dict()
    energy_curr = 10000
    for sa_step in range(SA_STEPS):
        energy.append(energy_curr)
        params_dict = step_params_dict(params_dict_old)
    
        N_AGENTS = 100
        try:
            agents, shares, centrality = run_agent_simulation(N_AGENTS, params_dict)
        except Exception as e:
            print(e)
            continue
        energy_new = calc_energy(agents, shares, sa_step, SA_STEPS, centrality)
        acceptance = acceptance_proba(energy_curr, energy_new, sa_step, SA_STEPS)
        if np.random.uniform() < acceptance:
            params_dict_old = params_dict
            energy_curr = energy_new
    rnd_info.append({'params_dict': params_dict_old, 'energy': energy})
    if rnd % 10 == 0:
        pickle.dump(rnd_info, open('rnd_info_{}.pkl'.format(str(rnd)), 'wb'))

pickle.dump(rnd_info, open('rnd_info_donezo.pkl', 'wb'))
