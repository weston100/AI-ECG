# AI-ECG
This repo contains tools for training, evaluating, and running deep learning models on electrocardiogram (ECG) waveforms. 

A paper with more details on this project will be published shortly. At that point, any work using this repo should cite that paper.

## Installation
First, clone this repository and enter the directory by running:
```
git clone https://github.com/weston100/AI-ECG.git
cd AI-ECG
```
Then, install dependencies using:
```
pip install -r requirements.txt
```
## Usage
### Training
Training a model for a specific task is as simple as 
1. Building a filelist csv with filenames, labels, and train/valid/test splits
2. Updating the key `task_cfg["task_name"]` in config.py with information about the task
3. Updating paths in `cfg` in config.py to point to your waveforms and where you'd like models written
4. Running
   ```
   python3 -c "import ecg; ecg.ecg(task='task_name')"
   ```
You can update default hyperparameters in `config.py`, or modify them on a given training run using
```
python3 -c "import ecg; ecg.ecg(task='task_name', cfg_updates={<your hparams here>})"
```
where cfg_updates is a nested dict whose schema is a partial copy of config.py's cfg, which will overwrite the given values.

Training will create a new directory in `cfg["optimizer"]["save_path"]` named based on passed hyperparameters, containing:
* `best.pt`: The best model checkpoint
* `valid_preds.csv`: The y and yh values for the validation set (used to compute the reported AUC)
* `all_preds.csv`: The y and yh values for all examples in the filelist
* `all_waveforms_preds.csv` The y values for all examples in the waveform directory
* Copies of the filelists for each split

### Running a pre-trained model
To run a pre-trained model, simply run
```
python3 -c "import ecg; ecg.ecg(mode='eval', task='task_name', eval_model_path='.', eval_model_name='best.pt')"
```
Ensure that in `config.py`, `cfg["dataloader"][""waveforms"]` is set to the path to your waveforms. By default, this will run the model on all waveforms in the directory; you can instead use a filelist by passing the argument `use_label_file_for_eval=True`. 

All our models are trained using waveforms which are lead-wise normalized, notch filtered, and wandering-baseline filtered. At runtime, you can apply these transformations to your waveforms by updating the hyperparamters:
```
{"dataloader": {"normalize_x": True, "notch_filter": True, "baseline_filter": True}}
```

Model weights can be downloaded from the releases pane.
