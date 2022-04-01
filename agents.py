class misinfoAgent(object):
    """
    Agent in a network that may or may not spread misinfo.
    Has parameters:
        - agent_id: should be an int. unique identifier for the agent.
        - neighbors: the agent's neighbors in the graph
        - forcefulness: the "forcefulness" with which the agent convinces others & holds onto its beliefs
        - share_propensity: the likelihood that this agent will share misinfo
        - misinfo_belief: the likelihood that this agent believes in misinfo
        - trust_stability: how stable this agent's trust in its neighbors is
    """
    def __init__(
        self,
        agent_id,
        neighbors,
        forcefulness,
        share_propensity,
        misinfo_belief,
        trust_stability,
    ):
        super().__init__()
        self.agent_id = agent_id
        self.neighbors = neighbors
        self.forcefulness = forcefulness
        self.share_propensity = share_propensity
        self.misinfo_belief = misinfo_belief
        self.trust_stability = trust_stability
