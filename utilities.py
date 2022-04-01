import random

import networkx as nx
import numpy as np
import powerlaw


def make_er_graph(p, N_AGENTS, agents, params_dict):
    """
    Given an edge probability p, number of agents N_AGENTS, list of agents agents, and params_dict,
    create an Erdos-Reyni model w/ N_AGENTS nodes, probability p of an edge existing
    Return a networkx graph & the revised list of agents.
    """
    G = nx.Graph()
    for i in range(N_AGENTS):
        G.add_node(i)
        for j in range(N_AGENTS):
            draw = np.random.uniform()
            if draw < p:
                G.add_edge(i, j)
                agents[i].neighbors[j] = np.log(
                    np.random.beta(params_dict["B1_NTRUST"], params_dict["B2_NTRUST"])
                )
                agents[j].neighbors[i] = np.log(
                    np.random.beta(params_dict["B1_NTRUST"], params_dict["B2_NTRUST"])
                )
    return G, agents


def make_configuration_model_graph(N_AGENTS, alpha, agents, params_dict):
    """
    Given a number of agents N_AGENTS, power law exponent alpha, list of agents agents, and params_dict,
    create a configuration model from a power law distribution w/ exponent alpha.
    Return a networkx graph & the revised list of agents.
    """
    powerlaw_maker = powerlaw.Power_Law(xmin=1, discrete=True, parameters=[alpha])
    degrees = powerlaw_maker.generate_random(N_AGENTS)
    stubs = []
    if np.sum(degrees) % 2 != 0:
        degrees[-1] = degrees[-1] + 1
    for deg, node in zip(degrees, range(N_AGENTS)):
        for d in range(int(deg)):
            stubs.append("{}_{}".format(str(node), str(deg)))
    random.shuffle(stubs)
    G = nx.Graph()
    for i, j in zip(stubs[0::2], stubs[1::2]):
        i_node = int(i.split("_")[0])
        j_node = int(j.split("_")[0])
        G.add_edge(i_node, j_node)
        agents[i_node].neighbors[j_node] = np.log(
            np.random.beta(params_dict["B1_NTRUST"], params_dict["B2_NTRUST"])
        )
        agents[j_node].neighbors[i_node] = np.log(
            np.random.beta(params_dict["B1_NTRUST"], params_dict["B2_NTRUST"])
        )
    for i in range(len(agents)):
        if i not in G:
            G.add_node(i)
    return G, agents


def make_powerlaw_cluster_graph(N_AGENTS, agents, p=0.05):
    m = int(N_AGENTS / 50)
    G = nx.powerlaw_cluster_graph(N_AGENTS, m, p)
    return G, agents


def markov_update_log(quantity, threshold, step, proba_decrease):
    """
    MCMC updater. Given a quantity (log), a threshold over which we choose to update,
    a step (how much we increase/decrease the quantity), and the probability of the quantity decreasing,
    return the altered quantity.
    """
    rand1 = np.random.uniform()
    if rand1 > threshold:
        rand2 = np.random.uniform()
        if rand2 > proba_decrease:
            return np.log(min(0.999999999, np.exp(quantity) + step))
        else:
            return np.log(max(np.exp(quantity) - step, 0.000000001))
    else:
        return quantity
