import argparse
from pathlib import Path

from pydantic import BaseModel
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 28 * 28
        hidden_size = self.input_size * 10
        output_size = 10
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, xb):
        xb = xb.reshape((-1, self.input_size))
        xb = self.fc1(xb)
        xb = F.relu(xb)
        xb = self.fc2(xb)
        return F.log_softmax(xb)


class DataLoaderConfig(BaseModel):
    num_workers: int = 0
    prefetch_factor: int | None = None


class TrainingRunConfig(BaseModel):
    batch_size: int = 1000
    test_batch_size: int = 1000
    epochs: int = 1
    lr: float = 1.0
    gamma: float = 0.7
    seed: int = 1
    data_loader: DataLoaderConfig | None = None
    use_gpu: bool = True
    shuffle_data: bool = True
    log_interval: int = 10

    def save(self, path: Path) -> None:
        with path.open("w") as outstream:
            yaml.safe_dump(self.model_dump(mode="json"), outstream)


def load_config(path: Path) -> TrainingRunConfig:
    with path.open() as instream:
        config = yaml.safe_load(instream)
        return TrainingRunConfig.model_validate(config)


def pick_device(config: TrainingRunConfig):
    if config.use_gpu:
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_data_loaders(config: TrainingRunConfig) -> tuple[DataLoader, DataLoader]:

    train_kwargs = {"batch_size": config.batch_size}
    test_kwargs = {"batch_size": config.test_batch_size}
    if config.data_loader:
        accel_kwargs = {
            "num_workers": config.data_loader.num_workers,
            "prefetch_factor": config.data_loader.prefetch_factor,
            "persistent_workers": True,
            "shuffle": True,
        }
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader: DataLoader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    return train_loader, test_loader


def train_model(config: TrainingRunConfig, log_dir: Path) -> None:
    device = pick_device(config)
    print(f"Using device: {device}")

    model = Net()
    model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)

    train_loader, test_loader = build_data_loaders(config)

    tensorboard_log = SummaryWriter(log_dir=(log_dir / "tensorboard"))

    def train_epoch(epoch: int):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            with record_function("minibatch"):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                if batch_idx % config.log_interval == 0:
                    with record_function("print info"):
                        print(
                            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                                epoch,
                                batch_idx * len(data),
                                len(train_loader.dataset),
                                100.0 * batch_idx / len(train_loader),
                                loss.item(),
                            )
                        )
        tensorboard_log.add_scalar(
            "Loss/train", total_loss / len(train_loader.dataset), epoch
        )

    def test_epoch():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    profile_schedule = schedule(wait=1, warmup=1, active=1, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
        schedule=profile_schedule,
    ) as prof:
        for epoch in range(1, config.epochs + 1):
            with record_function("train"):
                train_epoch(epoch)
            with record_function("test"):
                test_epoch()
            scheduler.step()
            prof.step()

    tensorboard_log.flush()
    config.save(log_dir / "config.yaml")
    prof.export_chrome_trace(str(log_dir / "profile.json"))


def make_logdir(root: Path) -> Path:
    root.mkdir(exist_ok=True)
    run_count = len(list(root.iterdir())) + 1
    dir = root / str(run_count)
    dir.mkdir()
    return dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, nargs="?", default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("mnist_runs"))

    args = parser.parse_args()

    log_dir = make_logdir(args.log_dir)
    config = load_config(args.config) if args.config else TrainingRunConfig()
    train_model(config, log_dir)


if __name__ == "__main__":
    main()
