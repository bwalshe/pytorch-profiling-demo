import argparse
from pathlib import Path
from typing import Callable
import torch
from torch.profiler import profile, ProfilerActivity, schedule

_factories: dict[str, Callable] = {}
_default_job = None


def job_factory(name: str, default=False):
    global _default_job
    if default:
        _default_job = name

    def register(fn):
        _factories[name] = fn

    return register


def list_factories() -> set[str]:
    return set(_factories.keys())


def make_scheduler(using_scheduler: bool, total_iterations: int):
    if using_scheduler:
        return schedule(wait=1, warmup=1, active=total_iterations - 2)


def trace_name(dir: Path, job_type: str, with_scheduler: bool, with_stack) -> str:
    name = f"trace_{job_type}"
    if with_scheduler:
        name = name + "_scheduler"
    if with_stack:
        name = name + "_stack"
    name += ".json"

    return str(dir / name)


def setup_job(job_type: str) -> Callable:
    if job_type is None:
        raise RuntimeError("No job type specified")
    if job_type not in list_factories():
        raise RuntimeError(f"Unknown job type {job_type}")

    device = torch.device("cuda:0")
    print(f"Using {device}")
    return _factories[job_type](device)


def trace_loop(
    job_type: str, with_scheduler: bool, with_stack: bool, output_path: Path
):
    iterations = 5
    job = setup_job(job_type)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    scheduler = make_scheduler(with_scheduler, iterations)
    with profile(
        activities=activities, schedule=scheduler, with_stack=with_stack
    ) as prof:
        for _ in range(iterations):
            z = job()
            prof.step()
    prof.export_chrome_trace(
        trace_name(output_path, job_type, with_scheduler, with_stack)
    )


def run_cli(output_path: Path):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job", default=_default_job, type=str, choices=list_factories()
    )
    parser.add_argument(
        "--schedule", type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--stack", type=bool, default=False, action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()

    output_path.mkdir(exist_ok=True, parents=True)
    trace_loop(args.job, args.schedule, args.stack, output_path)
