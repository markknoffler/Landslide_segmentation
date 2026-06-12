import torch


def positive_scale(param: torch.Tensor, max_log: float = 8.0) -> torch.Tensor:
    """Bounded positive material parameters — prevents exp() blow-up during training."""
    return torch.exp(torch.clamp(param, min=-8.0, max=max_log))
