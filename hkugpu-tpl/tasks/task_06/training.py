def compute_loss(policy, observations, actions, config):
    return policy(observations, actions)
