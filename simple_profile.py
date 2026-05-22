import argparse
from pathlib import Path
import torch

from trace_harness import job_factory, run_cli


@job_factory("linear", default=True)
def make_linear(device):
    size = 1000

    weights = torch.rand((size, size), device=device)
    bias = torch.rand((size, 1), device=device)
    xs = torch.rand((size, 1), device=device)

    return lambda: weights @ xs + bias


@job_factory("invert")
def make_invert(device):
    size = 1000
    x = torch.rand((size, size), device=device)

    return lambda: x.inverse()


@job_factory("yolo_invert")
def make_yolo_invert(device):
    size = 1000
    x = torch.rand((size, size), device=device)

    return lambda: torch.linalg.inv_ex(x)


if __name__ == "__main__":
    run_cli(Path("traces") / "simple")
