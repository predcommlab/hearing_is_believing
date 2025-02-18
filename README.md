# Hearing Is Believing: Experiments & analysis code
*In this repository, all code for experiments and analysis can be found to reproduce results from:*

```
Schneider, F., & Blank, H. (2025). Hearing Is Believing.
```

Please cite this work as:

```graphql
@article{
    Hearing is Believing,
    author = {Fabian Schneider, Helen Blank},
    title = {Hearing Is Beliveing},
    journal = {...}    
}
```

## 🚀 Getting started
### ❗️ Required: Installation (Python)
Make sure you have [anaconda](https://anaconda.org) installed. Open your terminal, navigate to your working directory for this project and create a fresh environment:

```bash
conda create -n "sempriors" python=3.10.9
conda activate sempriors
```

Next, install all requirements for the project:

```
conda install --yes --file requirements.txt
```

If you are on an intel machine, please also run:

```bash
pip install intel-numpy
pip install scikit-learn-intelex
```

### ❕ Optional: GPU Acceleration (Python)
Almost all scripts will allow you to specify GPU(s) to accelerate computations. If you would like to make use of this, please make sure to install an appropriate version of `PyTorch` that is compatible with your GPU.

For more information, please consult their [offical documentation](https://pytorch.org/get-started/locally/).

### ❗️ Required: Installation (R)
Please make sure to install R and, ideally, [RStudio](https://posit.co/download/rstudio-desktop/). Note that our machines run on `R version 4.0.3 (2020-10-10)`. Once installed, please open RStudio and install the requirements like so:

```R
install.packages(c('lme4', 'lmerTest', 'emmeans', 'DHARMa', 'ggplot2', 'viridis'))
```

### ❗️ Required: Downloading data
In your terminal, navigate to the top level of your working directory and download the data like so:

```bash
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

```bash
n_workers=n     How many jobs are we running in parallel?
backend=str     Which backend should be used (numpy/torch)?
device=str      If backend is torch, which device should be used (mps/cuda/cpu)?
```

These will dramatically influence computation times. Per default, all scripts will try to use torch with CUDA acceleration. **Note** that the default behaviour of these scripts is to distribute all available CUDA devices across your `n_workers`. If you would like only a specific device to be used, please specify which device should be used, e.g.: 

```bash
python my_script.py n_workers=2 device=torch backend=cuda:0
```

### 💡 Hint: Validate your installation
If you would like to fully replicate key results from our paper---specifically, refitting our models rather than using the provided solutions---you should now ensure that everything is running smoothly. To do so, please navigate to `./analysis/eeg/` in your terminal. Now, run:

```bash
conda activate sempriors
python
>>> import rsa
>>> rsa.analysis.estimators.unit_tests_TimeDelayed_trf()
>>> rsa.analysis.estimators.unit_tests_TimeDelayed_rec()
>>> rsa.analysis.estimators.unit_tests_Encoder()
>>> rsa.analysis.estimators.unit_tests_Temporal()
>>> rsa.analysis.estimators.unit_tests_SparseEncoder()
>>> rsa.analysis.estimators.unit_tests_Decoder()
>>> rsa.analysis.estimators.unit_tests_B2B()
```

For each of these, you should receive `True` as an output if tests are passed. Please repeat these tests for the `torch` backend like so, where you may optionally test a different device:

```bash
conda activate sempriors
python
>>> import rsa
>>> rsa.analysis.estimators.torch.unit_tests_TimeDelayed_trf(device = 'cpu')
>>> rsa.analysis.estimators.torch.unit_tests_TimeDelayed_rec(device = 'cpu')
>>> rsa.analysis.estimators.torch.unit_tests_Encoder(device = 'cpu')
>>> rsa.analysis.estimators.torch.unit_tests_Temporal(device = 'cpu')
>>> rsa.analysis.estimators.torch.unit_tests_SparseEncoderdevice = 'cpu')
>>> rsa.analysis.estimators.torch.unit_tests_Decoder(device = 'cpu')
>>> rsa.analysis.estimators.torch.unit_tests_B2B(device = 'cpu')
```

### ❕ Optional: Validation experiment
Navigate to `./analysis/validation/`. Open `./inference_validation.R`, set the appropriate working directory in `L17` and run the script. Results can be found in `./analysis/validation/results/items.csv`.

### ❕ Optional: Online experiment
Navigate to `./analysis/online/`. In your terminal, run:

```bash
python run_fe.py
```

Next, open a new jupyter notebook like so:

```bash
jupyter notebook
```

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

```bash
python audio_preprocess.py fs=200 fsn=200 method=spectrogram
```

#### ❗️ Required: Computing semantic priors
During the experiment, we computed real-time estimates of semantic priors that participants had learned. Because these are computed deterministically and they require a relatively decent amount of space, these are not included in OSF. Please run:

```bash
python subject_rtfe.py
python subject_rtfe.py -generalised
python subject_rtfe.py -unspecific
python subject_rtfe.py -generalised -unspecific
```

These estimates will be used in all further scripts.

#### ❕ Optional: Behavioural analysis (task one)
Aggregate behavioural data by running:

```bash
python group_beh_mt1.py
```

Next, open `inference_beh_mt1.R`, adjust the working directory in `L23` and run the script. Results are written to `/analysis/eeg/data/processed/beh/mt1/`.  These will later be used during figure creation.

#### ❗️ Required: Figure 1
By now, we may replicate figure one by opening a jupyter notebook:

```bash
jupyter notebook
```

and selecting `./fig1_design.ipynb`. Run all cells consecutively. Figures will be saved to `/analysis/eeg/figures/` as `png`, `svg`, and `pdf` files.

#### ❕ Optional: Stimulus reconstruction
If you would like to run stimulus reconstruction over morphs, please run:

```
python group_rsa_rec.py n_workers=2 backend=torch device=cuda
```

Make sure to adjust the parameters as per [performance hints](#-hint-performance). **Note** that you can also skip only this reconstruction step and move on from here: Once stimulus reconstruction has been completed (or skipped), you may run the inference script like so:

```bash
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

```bash
python group_rsa_enc_topk.py n_workers=2 backend=torch device=cuda
python inference_rsa_enc_topk.py
```

Results from top-k encoding will be available from `/analysis/eeg/data/results/encoding_topk_b0-m0-c0.pkl.gz`and `/analysis/eeg/data/results/reports/encoding_topk_b0-m0-c0.html`. In your report, check which number of k produced the best models. We know, of course, that this is going to be `k=19`, so let us now refit our original similarity encoders to make sure our results remain robust:

```bash
python group_rsa_enc.py n_workers=2 backend=torch device=cuda n_topk=19
python inference_rsa_enc.py n_topk=19
```

Check your results in `/analysis/eeg/data/results/encoding_b0-m0-c0-k19.pkl.gz` and `/analysis/eeg/data/results/reports/encoding_b0-m0-c0-k19.html`.

#### ❗️ Required: Figure 2
We now have all results available to reproduce figure two from the paper. If necessary, open your jupyter notebook

```bash
jupyter notebook
```

and select `./fig2_predictions.ipynb`. Run all cells consecutively and find the resulting figure in `/analysis/eeg/figures/` as `png`, `svg`, and `pdf`.


#### ❕ Optional: Single-trial EEG encoding (task one)
First, please extract the full LLM activations---because these are quite large, it is genuinely faster to quickly extract them yourself rather than having to download them. **Note** that we do provide subspace projections, so you may also choose to skip this. To do this, please run

```bash
python audio_w2v2.py device=cuda folder=narrative
python audio_w2v2.py device=cuda folder=morphed
python audio_w2v2.py device=cuda folder=clear
python audio_w2v2.py device=cuda folder=vocoded
```

Next, perform the subspace projection like so:

```bash
python audio_w2v2_pca.py n_features=5
python audio_w2v2_pca.py n_features=10
```

Finally, run the back-to-back decoders to evaluate all layers:

```bash
python audio_w2v2_selection.py n_workers=2 backend=torch device=cuda n_features=5
python audio_w2v2_selection.py n_workers=2 backend=torch device=cuda n_features=10
```

Once this is completed, we may run the single-trial encoding models like so:

```bash
python group_rerp_mt1.py n_workers=2 backend=torch device=cuda s_bsl=llm n_features=5 s_mod=inv
python group_rerp_mt1.py n_workers=2 backend=torch device=cuda s_bsl=llm n_features=5 s_mod=spc
python inference_rerp_mt1.py s_bsl=llm n_features=5 s_mod=inv
python inference_rerp_mt1.py s_bsl=llm n_features=5 s_mod=spc
```

You may check results in `/analysis/eeg/data/results/rerp-mt1-k5-z0-s0-b0-inv-llm5.pkl.gz`, `/analysis/eeg/data/results/rerp-mt1-k5-z0-s0-b0-spc-llm5.pkl.gz` and, of course, reports in `/analysis/eeg/data/results/reports/rerp-mt1-k5-z0-s0-b0-inv-llm5.html` and `/analysis/eeg/data/results/reports/rerp-mt1-k5-z0-s0-b0-spc-llm5.html`. From these reports, find the best models and coefficients and run:

```bash
python group_rerp_mt1_knockout.py n_workers=2 backend=torch device=cuda s_bsl=llm n_features=5 s_mod=inv n_mod=2 a_coefs=all
python group_rerp_mt1_knockout.py n_workers=2 backend=torch device=cuda s_bsl=llm n_features=5 s_mod=spc n_mod=2 a_coefs=all
python inference_rerp_mt1_knockout.py s_bsl=llm n_features=5 s_mod=inv n_mod=2 a_coefs=all
python inference_rerp_mt1_knockout.py s_bsl=llm n_features=5 s_mod=spc n_mod=2 a_coefs=all
```

Results from knock-outs are available from `/analysis/eeg/data/results/rerp-mt1-ko-n2-call-k5-z0-s0-b0-spc-llm5.pkl.gz`, `/analysis/eeg/data/results/rerp-mt1-ko-n2-call-k5-z0-s0-b0-inv-llm5.pkl.gz`, and reports are available from `/analysis/eeg/data/results/reports/rerp-mt1-ko-n2-call-k5-z0-s0-b0-inv-llm5.html` and `/analysis/eeg/data/results/reports/rerp-mt1-ko-n2-call-k5-z0-s0-b0-spc-llm5.html`.

To replicate the robustness of these findings in the higher subspace projection and with respect to target words, please run:

```bash
python group_rerp_mt1.py n_workers=2 backend=torch device=cuda s_bsl=llm n_features=10 s_mod=inv
python group_rerp_mt1.py n_workers=2 backend=torch device=cuda s_bsl=llm n_features=10 s_mod=spc
python inference_rerp_mt1.py s_bsl=llm n_features=10 s_mod=inv
python inference_rerp_mt1.py s_bsl=llm n_features=10 s_mod=spc

python group_rerp_mt1_knockout.py n_workers=2 backend=torch device=cuda s_bsl=llm n_features=10 s_mod=inv n_mod=2 a_coefs=all
python group_rerp_mt1_knockout.py n_workers=2 backend=torch device=cuda s_bsl=llm n_features=10 s_mod=spc n_mod=2 a_coefs=all
python inference_rerp_mt1_knockout.py s_bsl=llm n_features=10 s_mod=inv n_mod=2 a_coefs=all
python inference_rerp_mt1_knockout.py s_bsl=llm n_features=10 s_mod=spc n_mod=2 a_coefs=all
```

and:

```bash
python group_rerp_mt1.py n_workers=2 backend=torch device=cuda s_bsl=tar s_mod=inv
python group_rerp_mt1.py n_workers=2 backend=torch device=cuda s_bsl=tar s_mod=spc
python inference_rerp_mt1.py s_bsl=tar s_mod=inv
python inference_rerp_mt1.py s_bsl=tar s_mod=spc

python group_rerp_mt1_knockout.py n_workers=2 backend=torch device=cuda s_bsl=tar s_mod=inv n_mod=2 a_coefs=all
python group_rerp_mt1_knockout.py n_workers=2 backend=torch device=cuda s_bsl=tar s_mod=spc n_mod=2 a_coefs=all
python inference_rerp_mt1_knockout.py s_bsl=tar s_mod=inv n_mod=2 a_coefs=all
python inference_rerp_mt1_knockout.py s_bsl=tar s_mod=spc n_mod=2 a_coefs=all
```

#### ❗️ Required: Figure 3
We can now reproduce figure three from the paper. Again, open your jupyter notebook:

```bash
jupyter notebook
```

and select `./fig3_surprisal.ipynb`. Run all cells consecutively and find the figures in `/analysis/eeg/figures/` as `png`, `svg`, and `pdf`.

#### ❕ Optional: Modeling response times by congruency (task two)
First, aggregate all behavioural data from task two:

```
python group_beh_mt2.py
```

Next, open `/analysis/eeg/inference_beh_mt2.R`, adjust the working directory in `L23` and run the script. Outputs are available from `/analysis/eeg/data/processed/beh/mt2/` and will be used for plotting.

#### ❕ Optional: Single-trial EEG encoding (task two)
To replicate single-trial EEG encoding in task two, please run:

```bash
python group_rerp_mt2.py n_workers=2 backend=torch device=cuda b_con=1 b_inc=0 s_mod=inv
python group_rerp_mt2.py n_workers=2 backend=torch device=cuda b_con=0 b_inc=1 s_mod=inv
python group_rerp_mt2.py n_workers=2 backend=torch device=cuda b_con=1 b_inc=0 s_mod=spc
python group_rerp_mt2.py n_workers=2 backend=torch device=cuda b_con=0 b_inc=1 s_mod=spc
```

