# Hearing Is Believing: Experiments & analysis code
*In this repository, all code for experiments and analysis can be found to reproduce results from:*

```
Schneider, F., & Blank, H. (2025). Hearing Is Believing.
```

## 🚀 Getting started
### ❗️ Required: Installation (Python)
Make sure you have [anaconda](https://anaconda.org) installed. Open your terminal, navigate to your working directory for this project and create a new environment:
```
conda create -n "sempriors" python=3.10.9
conda activate sempriors
```
Next, install all requirements for the project:
```
conda install --yes --file requirements.txt
```
If you are on an intel machine, please also run:
```
pip install intel-numpy
pip install scikit-learn-intelex
```

### ❕ Optional: GPU Acceleration (Python)
Almost all scripts will allow you to specify GPU(s) to accelerate computations. If you would like to make use of this, please make sure to install an appropriate version of `PyTorch` that is compatible with your GPU.

For more information, please consult their [offical documentation](https://pytorch.org/get-started/locally/).

### ❗️ Required: Installation (R)
Please make sure to install R and, ideally, [RStudio](https://posit.co/download/rstudio-desktop/). Note that our machines run on `R version 4.0.3 (2020-10-10)`. Once installed, please open RStudio and install the requirements like so:
```
install.packages(c('lme4', 'lmerTest', 'emmeans', 'DHARMa', 'ggplot2', 'viridis'))
```

### ❗️ Required: Downloading data
In your terminal, navigate to the top level of your working directory and download the data like so:
```
pip install osfclient
osf init
osf -p ctrma fetch -r / ./
```
Upon the init command, you may be prompted to input your OSF account and the project id, which is `ctrma`. Downloading all data may take a while, as the project is about `50GB`. Please verify you have enough space before trying to download the data.

Alternatively, use your browser to navigate to [https://osf.io/ctrma](https://osf.io/ctrma) and download the full zip. Make sure to extract it to the top-level directory.

## 🗂️ Project structure
Your repository should now look roughly like this:
```
/root
│── /experiments/           # Source code for experiments
|──── /validation/          # Validation experiment
|────── /data/
|────── /resources/ 
|──── /online/              # Online experiment
|────── /data/
|────── /resources/ 
|──── /eeg/                 # EEG experiment
|────── /models/
|────── /resources/
|────── /rtfe/
|── /analysis/              # Source code for analysis
|──── /validation/          # Validation experiment
|────── /data/
|──── /online/              # Online experiment
|────── /data/
|──── /eeg/                 # EEG experiment
|────── /data/
|────── /rsa/
|──── /spaces/                      # Common packages and embeddings
|────── /pubplot/                   # Plotting utilities
|────── /embeddings/                # Embedding utilities
|────── /dumps/                     # Descriptive files for stimuli
|────── /glove-german/              # GloVe embeddings
|────── /text-embedding-ada-002/    # GPT3 embeddings
|────── /llama-7b/                  # LLama7b embeddings
|────── /bert-base-german-cased/    # BERT embeddings
│── README.md               # This file
```

## ⚙️ Usage
Here, we will walk through the basic steps of replicating all analyses reported in the paper and supplement. Not all steps that we report here will be necessary, if all you want to do is replicate the figures from our paper, as all processed data are included in OSF. This is because EEG analysis in particular is computationally expensive. For reference, we used two `NVIDIA RTX 5000 Ada` GPUs and the full analysis took about three weeks to run. Therefore, you may skip all optional steps if you would like to reproduce figures from processed results.

### 💡 Hint: Performance
Almost all computationally expensive modelling is done in the python scripts. For all of these, you may specify three key settings in your calls:
```
n_workers=n     How many jobs are we running in parallel?
backend=str     Which backend should be used (numpy/torch)?
device=str      If backend is torch, which device should be used (mps/cuda/cpu)?
```
These will dramatically influence computation times. Per default, all scripts will try to use torch with CUDA acceleration. **Note** that the default behaviour of these scripts is to distribute all available CUDA devices across your `n_workers`. If you would like only a specific device to be used, please specify which device should be used, e.g.: 

```python my_script.py n_workers=2 device=torch backend=cuda:0```

### ❕ Optional: Validation experiment
Navigate to `./analysis/validation/`. Open `./inference_validation.R`, set the appropriate working directory in `L17` and run the script. Results can be found in `./analysis/validation/results/items.csv`.

### ❕ Optional: Online experiment
Navigate to `./analysis/online/`. In your terminal, run:

```python run_fe.py```

Next, open a new jupyter notebook like so:

```jupyter notebook```

and open `data.ipynb` and follow the cells to aggregate all the data. Once you have run all cells, you may close the notebook and kernel. In RStudio, open`./run_glmmm.R` and adjust your working directory in `L24`. Run the code. Results will be available from `./analysis/online/results/glmm/`.

### ❗️ Required: EEG experiment
This is where all analyses will come together and where figures will be created, hence I have marked this section as required. However, as mentioned above, many of these subsections will be computationally expensive. Individual sections that *must* be run will be marked again. Please make sure to navigate to `./analysis/eeg/`.

#### 💡 Hint: File names
For EEG analysis, files come in essentially four variants. These are:

```
/analysis/eeg/
|── audio_*.py          # Files associated with preprocessing audio data
|── subject_*.py        # Subject-level analysis scripts
|── group_*.py          # Group-level aggregation or analysis
|── inference_*         # Group-level statistical inference
```

Whenever you are interested about available parameters of or the inner workings of some kind of analysis, you will generally want to refer to the corresponding `subject_* .py` file that will contain descriptions of all parameters, usage notes, and the full code and documentation. Note that almost all `group_*` files are convenience scripts that will call individual `subject_*` scripts for all subjects.

#### ❕ Optional: Computing gammatone spectrograms
Gammatone spectrograms for all stimuli can be extracted by running:
```
python audio_preprocess.py fs=200 fsn=200 method=spectrogram
```

#### ❗️ Required: Computing semantic priors
During the experiment, we computed real-time estimates of semantic priors that participants had learned. Because these are computed deterministically and they require a relatively decent amount of space, these are not included in OSF. Please run:
```
python subject_rtfe.py
python subject_rtfe.py -generalised
python subject_rtfe.py -unspecific
python subject_rtfe.py -generalised -unspecific
```
These estimates will be used in all further scripts.

#### ❕ Optional: Behavioural analysis (task one)
Aggregate behavioural data by running:

```
python group_beh_mt1.py
```

Next, open `inference_beh_mt1.R`, adjust the working directory in `L23` and run the script. Results are written to `/analysis/eeg/data/processed/beh/mt1/`.  These will later be used during figure creation.

#### ❗️ Required: Figure 1
By now, we may replicate figure one by opening a jupyter notebook:

```
jupyter notebook
```

and selecting `./fig1_design.ipynb`. Run all cells consecutively. Figures will be saved to `/analysis/eeg/figures/` as `png`, `svg`, and `pdf` files.

#### ❕ Optional: Stimulus reconstruction
If you would like to run stimulus reconstruction over morphs, please run:

```
python group_rsa_rec.py n_workers=2 backend=torch device=cuda
```

Make sure to adjust the parameters as per [performance hints](#-hint-performance). **Note** that you can also skip only this reconstruction step and move on from here: Once stimulus reconstruction has been completed (or skipped), you may run the inference script like so:

```
python inference_rsa_rec.py
```

Running this will compute all relevant statistics and tests. These will be available from `/analysis/eeg/data/results/reconstruction.pkl.gz`. Additionally, the script will generate an `MNE` report summarising key results that is available from `/analysis/eeg/data/results/reports/reconstruction.html`.

#### ❕ Optional: Similarity encoding
If you would like to run similarity encoders from scratch, please run the following script, adjusting parameters [as necessary](#-hint-performance):

```
python group_rsa_enc.py n_workers=2 backend=torch device=cuda
```

Results from encoders will be available from `/analysis/eeg/data/results/encoding_b0-m0-c0-k5.pkl.gz` and, for the report, `/analysis/eeg/data/results/reports/encoding_b0-m0-c0-k5.html`.

Next, you may rerun encoders while systematically varying the number of top-k predictions considered like so:

```
python group_rsa_enc_topk.py n_workers=2 backend=torch device=cuda
```

Results from top-k encoding will be available from `/analysis/eeg/data/results/encoding_topk_b0-m0-c0.pkl.gz`and `/analysis/eeg/data/results/reports/encoding_topk_b0-m0-c0.html`. In your report, check which number of k produced the best models. We know, of course, that this is going to be `k=19`, so let us now refit our original similarity encoders to make sure our results remain robust:

```
python group_rsa_enc.py n_workers=2 backend=torch device=cuda n_topk=19
```

Check your results in `/analysis/eeg/data/results/encoding_b0-m0-c0-k19.pkl.gz` and `/analysis/eeg/data/results/reports/encoding_b0-m0-c0-k19.html`.

#### ❗️ Required: Figure 2
We now have all 