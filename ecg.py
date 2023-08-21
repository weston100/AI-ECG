"""A function for training and evaluating ECG models."""

import copy
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import torch

import config
from dataset import ECGDataset
from train import Trainer, optimizer_and_scheduler
from model import ECGModel

def ecg(
        mode="train",
        task="cvm",
        eval_model_path=".",
        eval_model_name="best.pt",
        use_label_file_for_eval=False,
        cfg_updates={},
        log_path="",
    ):
    """
    A function to train and evaluate ECG models.
    Arguments:
        mode: Either "train" or "eval".
        task: The task to train or eval on. Should be a key in config.py's task_cfg.
        eval_model_path: Only used in eval mode. Path to the directory for evaluation.
        eval_model_name: Only used in eval mode. Name of the model to evaluate.
        use_label_file_for_eval: Only used in eval mode. If true, load a label file;
            if false, evaluate on all files in a directory.
        cfg_updates: A nested dict whose schema is a partial copy of config.py's cfg.
            This dict will be used to overwrite the elements in cfg, and to name directories
            during training.
        log_path: Mainly for use with Stanford infrastructure. If set, the file at this path
            will be copied to the model directory after training/evaluation.

    For training, set `task` and overwrite any hyperparameters in `cfg_updates`.

    For evaluation, set `mode="eval"`, set `task` and overwrite any hyperparameters in `cfg_updates`,
    and set `eval_model_path`, `eval_model_name`, and `use_label_file_for_eval` as necessary.
    """

    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = config.update_config_dict(
        config.cfg,
        config.task_cfg[task],
    )
    cfg = config.update_config_dict(
        config.cfg,
        cfg_updates,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    model_name = f"{task},{config.dict_to_str(cfg_updates)}"
    print(model_name, flush=True)
    if mode == "train":
        output = os.path.join(cfg["optimizer"]["save_path"], model_name)
        os.makedirs(output, exist_ok=True)
    else:
        output = eval_model_path 

    model = ECGModel(
        cfg["model"],
        num_input_channels=(len(cfg["dataloader"]["leads"])
            if cfg["dataloader"]["leads"] else 12),
        num_outputs=len(cfg["dataloader"]["label_keys"]),
        binary=cfg["dataloader"]["binary"]
    ).float()

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    optim, scheduler = optimizer_and_scheduler(cfg["optimizer"], model)

    if mode == "train" or use_label_file_for_eval:
        datasets = {k: ECGDataset(cfg["dataloader"], k, output=output) for k in ["train", "valid", "test", "all"]}
        dataloaders = {
            key:
                torch.utils.data.DataLoader(
                    datasets[key],
                    batch_size=cfg["optimizer"]["batch_size"],
                    num_workers=cfg["dataloader"]["n_dataloader_workers"],
                    shuffle=(key == "train"),
                    drop_last=(key == "train"),
                    pin_memory=True,
                )
            for key in ["train", "valid", "test", "all"]}
    else:
        datasets, dataloaders = None, None

    trainer = Trainer(
                cfg,
                device,
                model,
                optim,
                scheduler,
                datasets,
                dataloaders,
                output,
    )

    if mode == "train":
        trainer.train()

    elif mode == "eval":
        best_epoch, best_score = trainer.try_to_load("best.pt")
        print(f"Best score seen: {best_score:.3f} at epoch {best_epoch}", flush=True)
        if use_label_file_for_eval:
            trainer.run_eval_on_split("test", report_performance=True)
        else:
            trainer.run_eval_on_all()

    if log_path:
        shutil.copy(log_path, output)

