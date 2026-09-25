from torch.utils.data import Dataset


class TaskWindows(Dataset):
    """Return (observation window, action prediction window) tensors per item."""

    def __init__(self, episodes, config):
        # TODO: Build episode-safe temporal windows and terminal padding here.
        raise NotImplementedError("Implement TaskWindows for your demonstrations")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


def load_datasets(path, seed, validation_fraction, limit, config):
    """Load, preprocess, and split by episode; return train and validation datasets."""
    # TODO: Decode your dataset format and create TaskWindows for both splits.
    # Keep validation episodes separate from training episodes to avoid leakage.
    raise NotImplementedError("Implement the task-specific data pipeline")
