"""A dataset for loading ECG data."""

import copy
import os
import glob
import tqdm

import numpy as np
import pandas as pd
import torch
import scipy.signal

class ECGDataset(torch.utils.data.Dataset):
    """
    A torch dataset of ECG examples.

    Arguments:
        cfg: The config dict, like config.py's cfg["dataloader"].
        split: The name of the split.
        output: The output directory.
        return_labels: Whether to return labels with waveforms. Useful to turn off
            during evaluation.
        all_waveforms: Whether to use all files in a directory, rather than using a label file.

    Calling Trainer.train() will train a full model; calling Trainer.run_eval_on_split()
    and Trainer.run_eval_on_all() evaluates the model on a list of files and a directory
    of files.
    """
    def __init__(
        self,
        cfg,
        split="train",
        output=None,
        return_labels=True,
        all_waveforms=False):

        self.cfg = cfg
        self.split = split
        self.output = output
        self.return_labels = return_labels and not all_waveforms
        self.all_waveforms = all_waveforms

        self.load_filelist()
        if self.return_labels:
            for label in cfg["label_keys"]:
                self.load_label(label)
        self.save_filelist()
        self.get_mean_and_std()
        self.filenames = self.filelist[cfg["filekey"]]
        print(self.filelist, flush=True)

    def __getitem__(self, index):
        waveform = self.get_waveform(index)
        if self.return_labels:
            y = self.filelist[self.cfg["label_keys"]].loc[[index]].to_numpy().flatten()
            y = torch.from_numpy(y).float()
            return [waveform, y]
        else:
            return [waveform]

    def __len__(self):
        return len(self.filelist)

    def load_filelist(self):
        if self.all_waveforms:
            print("Running on all files", flush=True)
            self.filelist = pd.DataFrame(
                    {self.cfg["filekey"]: [s.split("/")[-1] for s in glob.glob(os.path.join(self.cfg["waveforms"], "*"))],
                     "split": "all"
                    })
        else:
            label_csv = self.cfg["label_file"]
            print(f"Loading from {label_csv}", flush=True)
            self.filelist = pd.read_csv(label_csv)

        print(f"{len(self.filelist)} files in list", flush=True)

        if self.split != "all":
            if self.cfg["crossval_idx"]:
                self.filelist["split"][self.filelist["split"] == "valid"] = "train"
                self.filelist["split"][self.filelist["split"] == "train{}".format(cfg["crossval_idx"])] = "valid"
            self.filelist["split"][self.filelist["split"].str.contains("train")] = "train"

            self.filelist = self.filelist[self.filelist["split"] == self.split]
        print(f"{len(self.filelist)} files in split {self.split}", flush=True)

        if self.cfg["remove_labels"] and self.return_labels:
            overreads = pd.read_csv(self.cfg["overread_csv"])
            self.filelist = self.filelist.merge(overreads, how="left", on=self.cfg["filekey"], suffixes=("", "_y"))
            print(f"{len(self.filelist)} files after merging with overreads", flush=True)
            for remove_label in self.cfg["remove_labels"]:
                self.filelist = self.filelist[~self.filelist[remove_label].fillna(False)]
        print(f"{len(self.filelist)} files without removal criteria", flush=True)

        self.filelist.reset_index(drop=True, inplace=True)

    def load_label(self, label):
        for label in self.cfg["label_keys"]:
            if self.cfg["binary"]:
                if self.cfg["binary_positive_class"] == "below":
                    self.filelist[label] = self.filelist[label] <= self.cfg["binary_cutoff"]
                elif self.cfg["binary_positive_class"] == "above":
                    self.filelist[label] = self.filelist[label] >= self.cfg["binary_cutoff"]
                print(f"{self.filelist[label].sum()} positive examples in class {label}", flush=True)
            else:
                if self.cfg["normalize_y"]:
                    self.filelist[label] = (
                        self.filelist[label] - np.mean(self.filelist[label])
                        ) / np.std(self.filelist[label])
                print(f"Mean {self.filelist[label].mean()}, std {self.filelist[label].std()} in class {label}",
                    flush=True)

    def save_filelist(self):
        if self.output:
            fname = os.path.join(self.output, "{}_filelist.csv".format(self.split))
            if not os.path.exists(fname):
                self.filelist.to_csv(fname)

    def get_waveform(self, index):
        f = self.filelist[self.cfg["filekey"]][index]
        if "." not in f:
            f = f"{f}.{self.cfg['waveform_type']}"
        f = os.path.join(self.cfg["waveforms"], f)
        x = np.load(f).astype(np.float32)

        if len(x) <= 12:
            x = x.T
        if len(x) > 5000:
            x = x[:5000]
        if self.cfg["notch_filter"]:
            x = self.notch(x)
        if self.cfg["baseline_filter"]:
            x = self.baseline(x)
        if self.cfg["downsample"]:
            x = x[::self.cfg["downsample"]]
            expected_length = 5000 // self.cfg["downsample"]
        else:
            expected_length = 5000

        assert len(x) >= expected_length, (
            f"Got signal of length {len(x)}, which is too short for expected_length {expected_length}")
        assert (len(x) < 2 * expected_length) and not self.cfg["accept_all_lengths"], (
            f"Got signal of length {len(x)}, which looks too long for expected_length {expected_length}")
        # Okay to crop 5300 -> 5000, don"t want to crop 5000 -> 2500 as this looks like a sampling problem
        x = x[:expected_length]

        if self.cfg["normalize_x"]:
            x = (x - self.cfg["x_mean"][:x.shape[1]]) / self.cfg["x_std"][:x.shape[1]]
        x[np.isnan(x)] = 0
        x[x == np.inf] = x[x != np.inf].max()
        x[x == -np.inf] = x[x != -np.inf].min()
        if self.cfg["leads"]:
            x = x[:, self.cfg["leads"]]
        x = x.T

        return torch.from_numpy(x).float()

    def notch(self, data):
        data = data.T
        upsample = 5000 // data.shape[1]
        sampling_frequency = 500
        row, __ = data.shape
        processed_data = np.zeros(data.shape)
        b = np.ones(int(0.02 * sampling_frequency)) / 50.
        a = [1]
        for lead in range(0, row):
            if upsample and upsample != 1:
                X = scipy.signal.resample(data[lead, :], 5000)
                X = scipy.signal.filtfilt(b, a, X)
                X = X[::upsample]
            else:
                X = scipy.signal.filtfilt(b, a, data[lead,:])
            processed_data[lead,:] = X
        return processed_data.T

    def baseline(self, data):
        data = data.T
        row,__ = data.shape
        sampling_frequency = data.shape[1] // 10

        win_size = int(np.round(0.2 * sampling_frequency)) + 1
        baseline = scipy.ndimage.median_filter(data, [1, win_size], mode="constant")
        win_size = int(np.round(0.6 * sampling_frequency)) + 1
        baseline = scipy.ndimage.median_filter(baseline, [1, win_size], mode="constant")
        filt_data = data - baseline

        return filt_data.T

    def get_mean_and_std(self, batch_size=128, samples=8192):
        if ("x_mean" in self.cfg and "x_std" in self.cfg) or not self.cfg["normalize_x"]:
            return
        cfg = copy.deepcopy(self.cfg)
        self.cfg["return_labels"], self.cfg["normalize_x"] =  False, False

        indices = np.random.choice(len(self), min(len(self), samples), replace=False)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(self, indices),
            batch_size=batch_size,
            num_workers=self.cfg["n_dataloader_workers"],
            shuffle=False
        )

        n = 0 
        s1 = 0.
        s2 = 0.

        print("loading mean and std", flush=True)
        for x in tqdm.tqdm(dataloader):
            x = x[0].transpose(0, 1).reshape(x[0].shape[1], -1) #channels, -1
            n += np.float64(x.shape[1])
            s1 += torch.sum(x, dim=1).numpy().astype(np.float64)
            s2 += torch.sum(x ** 2, dim=1).numpy().astype(np.float64)
        x_mean = (s1 / n).astype(np.float32)
        x_std = np.sqrt(s2 / n - x_mean ** 2).astype(np.float32)

        if n < samples:
            print(f"WARNING: calculating mean and std based on {n} waveforms", flush=True)

        print(f"Means: {x_mean}")
        print(f"Stds: {x_std}")

        self.cfg = cfg
        self.cfg["x_mean"] = x_mean
        self.cfg["x_std"] = x_std
        
