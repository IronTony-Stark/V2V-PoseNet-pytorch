import torch
import torch.optim as optim
from lib.progressbar import progress_bar
from tqdm import tqdm


def train_epoch(model, criterion, optimizer, train_loader, epoch,
                device=torch.device('cuda'), dtype=torch.float, wandb_run=None):
    model.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in tqdm(
            enumerate(train_loader), total=len(train_loader), desc="Batches", leave=False):
        inputs, targets = inputs.to(device, dtype), targets.to(device, dtype)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if wandb_run:
            wandb_run.log({"train": {
                "loss": train_loss / (batch_idx + 1),
            }})


def val_epoch(model, criterion, val_loader, epoch,
              device=torch.device('cuda'), dtype=torch.float, wandb_run=None):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(
                enumerate(val_loader), total=len(val_loader), desc="Batches", leave=False):
            inputs, targets = inputs.to(device, dtype), targets.to(device, dtype)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()

            if wandb_run:
                wandb_run.log({"val": {
                    "loss": val_loss / (batch_idx + 1),
                }})


def test_epoch(model, test_loader, result_collector,
               device=torch.device('cuda'), dtype=torch.float, wandb_run=None):
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, extra) in tqdm(
                enumerate(test_loader), total=len(test_loader), desc="Batches", leave=False):
            outputs = model(inputs.to(device, dtype))
            result_collector((inputs, outputs, extra))
