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

```python my_script.py n_workers=2 device=torch backend=cuda:0```.

### ❕ Optional: Validation experiment
