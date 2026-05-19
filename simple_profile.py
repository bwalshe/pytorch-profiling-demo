import argparse
from pathlib import Path
import torch
from torch.profiler import record_function, profile, ProfilerActivity, schedule


def make_scheduler(using_scheduler: bool, total_iterations: int):
    if using_scheduler:
        return schedule(wait=1, warmup=1, active=total_iterations - 2)


def trace_name(dir: Path, with_scheduler: bool, with_stack) -> str:
    name = "simple_trace"
    if with_scheduler:
        name = name + "_scheduler"
    if with_stack:
        name = name + "_stack"
    name += ".json"

    return str(dir / name)


def main(with_scheduler: bool, with_stack: bool, output_path: Path):
    size = 1000
    iterations = 5
    device = torch.device("cuda:0")
    print(f"Using {device}")

    weights = torch.ones((size, size), device=device)
    bias = torch.ones((size, 1), device=device)

    xs = torch.ones((size, 1), device=device)

    total = 0
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    scheduler = make_scheduler(with_scheduler, iterations)
    with profile(
        activities=activities, schedule=scheduler, with_stack=with_stack
    ) as prof:
        for _ in range(5):
            with record_function("liniar"):
                zs = weights @ xs + bias
            total += zs.sum().item()
            prof.step()
    prof.export_chrome_trace(trace_name(output_path, with_scheduler, with_stack))
    print(f"total = {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule", type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--stack", type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()

    trace_dir = Path("traces")
    trace_dir.mkdir(exist_ok=True)
    main(args.schedule, args.stack, trace_dir)
