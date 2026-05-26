from pathlib import Path
import torch

from trace_harness import job_factory, run_cli


def make_data(device):
    size = 1000
    a = torch.rand((size, size), device=device)
    b = torch.rand((size, size), device=device)
    return a, b


@job_factory("invert", default=True)
def make_invert(device):
    a, b = make_data(device)

    def job():
        with torch.Stream(device):
            a_inv = a.inverse()
        with torch.Stream(device):
            b_inv = b.inverse()
        x = a_inv @ b_inv

    return job


@job_factory("invert_ex")
def make_invert_ex(device):
    a, b = make_data(device)

    def job():
        with torch.Stream(device):
            a_inv, _ = torch.linalg.inv_ex(a)
        with torch.Stream(device):
            b_inv, _ = torch.linalg.inv_ex(b)
        a_inv @ b_inv

    return job


@job_factory("invert_algebraic")
def make_invert_alg(device):
    a, b = make_data(device)

    return lambda: (b @ a).inverse()


if __name__ == "__main__":
    run_cli(Path("traces") / "multi_stream")
