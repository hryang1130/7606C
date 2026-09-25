def compute_loss(policy, observations, actions, config):
    """One optimizer update's scalar objective; edit for your algorithm."""
    # TODO: Implement the method-specific training objective, e.g. noise-prediction loss.
    # policy is the object returned by policy.build_policy(...).
    return policy(observations, actions)
