import torch


def make_eval_env(config, num_envs, backend):
    # TODO: Create the GPU-vectorized ManiSkill environment, frame stack, and vector wrapper.
    raise NotImplementedError


def make_video_env(config, output_dir):
    # TODO: Create a rendered single environment wrapped by ManiSkill RecordEpisode.
    raise NotImplementedError


def observation_tensor(observation, device):
    # TODO: Flatten/normalize task observations and place them on the policy device.
    return observation.to(device) if torch.is_tensor(observation) else torch.as_tensor(observation, device=device)


def video_observation(observation, device):
    # TODO: Return one policy observation with a batch dimension.
    return observation_tensor(observation, device).unsqueeze(0)


def select_action_chunk(prediction, config):
    # TODO: Select the action horizon that should be executed from each prediction.
    raise NotImplementedError


def success_mask(info, num_envs, device):
    # TODO: Convert the task success signal to a bool tensor of shape [num_envs].
    raise NotImplementedError


def video_action(action):
    # TODO: Convert a single policy action to the shape/type accepted by env.step.
    return action.detach().cpu().numpy()
