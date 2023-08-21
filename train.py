"""A trainer for ECG deep learning models."""

import os
import tqdm

from dataset import ECGDataset

import numpy as np
import pandas as pd
import torch

class Trainer:
    """
    A training object for ECG deep learning models.

    Arguments:
        cfg: The config dict, like config.py's cfg.
        device: A torch.device.
        model: A model, as defined in model.py.
        optim: A torch optimizer.
        scheduler: A torch scheduler.
        datasets: A dict of torch datasets for each split.
        dataloaders: A dict of torch dataloaders for each split.
        output: The output directory.

    Calling Trainer.train() will train a full model; calling Trainer.run_eval_on_split()
    and Trainer.run_eval_on_all() evaluates the model on a list of files and a directory
    of files.
    """
    def __init__(
        self,
        cfg,
        device,
        model,
        optim,
        scheduler,
        datasets,
        dataloaders,
        output,):
        self.cfg = cfg
        self.device = device
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.output = output

    def train(self):
        n_epochs = self.cfg["optimizer"]["n_epochs"]
        epoch_resume, best_score = self.try_to_load()

        if (self.cfg["optimizer"]["reduce_on_plateau"] and 
            epoch_resume and
            self.cfg["optimizer"]["lr"] * self.cfg["optimizer"]["max_reduction"] 
                > self.optim.param_groups[0]["lr"]):
            epoch_resume = n_epochs        

        for epoch in range(epoch_resume, n_epochs):
            for split in ["train", "valid"]:             
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_max_memory_allocated(i)
                    torch.cuda.reset_max_memory_cached(i)
                print(f"epoch {epoch} {split}")
                losses, ys, yhs, score = self.run_epoch(split)
                print(f"{epoch} {split}, {score:.3f}", flush=True)

            plateaued = False
            if self.cfg["optimizer"]["reduce_on_plateau"]:
                self.scheduler.step(np.mean(losses))
                print("num bad epochs", self.scheduler.num_bad_epochs, flush=True)
                print("lr", self.optim.param_groups[0]["lr"], flush=True)
                if (self.cfg["optimizer"]["lr"] * self.cfg["optimizer"]["max_reduction"]
                        > self.optim.param_groups[0]["lr"]):
                    print("plateaued", flush=True)
                    plateaued = True
            else:
                self.scheduler.step()
            best_score = self.save_model(losses, epoch, score, best_score)
            if plateaued:
                break

        best_epoch, best_score = self.try_to_load("best.pt")
        print(f"Best score seen: {best_score:.3f} at epoch {best_epoch}", flush=True)
        self.run_eval_on_split("valid", report_performance=True)
        self.run_eval_on_split("all")
        self.run_eval_on_all()


    def run_epoch(self, split, no_label=False, dataloader=None):
        if not dataloader:
            print(f"Loading dataloader for split {split}", flush=True)
            dataloader = self.dataloaders[split]

        if split == "train":
            self.model.train()
        else:
            self.model.eval()

        losses, ys, yhs = [], [], []
        for i, D in enumerate(tqdm.tqdm(dataloader)):
            if no_label:
                x = D[0].to(self.device)
                yh = self.model.module.forward(x)
                yhs.extend(yh.data.tolist())
            else:
                x = D[0].to(self.device)
                y = D[1].to(self.device)
                (y, yh, loss) = self.model.module.train_step(x, y)
                assert not np.isnan(loss.item()), "loss is nan at step {}".format(i)
                if split=="train":
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                losses.append(loss.item())
                ys.extend(y.data.tolist())
                yhs.extend(yh.data.tolist())  

        ys, yhs, losses = np.array(ys), np.array(yhs), np.array(losses)

        if no_label:
            score = 0
        else:
            score = self.model.module.score(ys, yhs, np.mean(losses))
        return losses, ys, yhs, score


    def run_eval_on_split(self, split, report_performance=False):
        print(f"Running eval on {split}")
        losses, ys, yhs, _ = self.run_epoch(split)
        np.save(os.path.join(self.output, f"{split}_names.npy"), self.datasets[split].filenames)
        pd.DataFrame({"fname": self.datasets[split].filenames,
                        "y": ys.flatten(),
                        "yh": yhs.flatten()}).to_csv(os.path.join(self.output, f"{split}_preds.csv"))
        if report_performance:
            score = self.model.module.score(ys, yhs, losses)
            print(f"{split} score: {score:.3f}")


    def run_eval_on_all(self):
        dataset = ECGDataset(self.cfg["dataloader"],
                             split="all",
                             all_waveforms=True)
        dataloader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.cfg["optimizer"]["batch_size"],
                            shuffle=False,
                            num_workers=self.cfg["dataloader"]["n_dataloader_workers"],
                            pin_memory=True,
                            drop_last=False)
        print("running eval on all", flush=True)
        _, _, yhs, _ = self.run_epoch("all", no_label=True, dataloader=dataloader)
        pd.DataFrame({"fname": dataset.filenames, "yh": yhs.flatten()}).to_csv(
            os.path.join(self.output, "all_waveforms_preds.csv"))

    def try_to_load(self, name="checkpoint.pt", model_path=None):
        try:
            if not model_path:
                model_path = os.path.join(self.output, name)
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optim.load_state_dict(checkpoint["opt_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_dict"])
            epoch_resume = checkpoint["epoch"] + 1
            best_score = checkpoint["best_score"]
            print(f"Resuming from epoch {epoch_resume}")
        except FileNotFoundError:
            print("Starting run from scratch")
            epoch_resume = 0
            best_score = 0
        return epoch_resume, best_score


    def save_model(self, losses, epoch, score, best_score, save_all=False):
        loss = np.mean(losses)
        best = False
        if best_score < score:
            best_score = score
            best = True

        # Save checkpoint
        save = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_score": best_score,
            "loss": loss,
            "opt_dict": self.optim.state_dict(),
            "scheduler_dict": self.scheduler.state_dict(),
        }

        torch.save(save, os.path.join(self.output, "checkpoint.pt"))
        if best:
            torch.save(save, os.path.join(self.output, "best.pt"))
        if save_all:
            torch.save(save, os.path.join(self.output, "checkpoint{}.pt".format(epoch)))
        return best_score


def optimizer_and_scheduler(cfg, model):
    """
    A function to build a torch optimizer and scheduler.
    Arguments:
        cfg: The config dict, like config.py's cfg["optimizer"].
        model: The torch model to train.
    """
    if cfg["optimizer"] == "adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"])
    else:
        optim = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=.9,
            weight_decay=cfg["weight_decay"])

    if cfg["reduce_on_plateau"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=cfg["patience"])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, cfg["n_epochs"] / cfg["lr_plateaus"])
    return optim, scheduler
