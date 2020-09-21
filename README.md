# Predicting Accidents in the Mining Industry with a Multi-Task Learning Approach

> A multi-task learning approach to train the mining accident risk prediction models.

![Python 3.x](https://img.shields.io/badge/python-3.x-green.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Inria-Chile/mining-risk-multitask-model)
![CI](https://github.com/Inria-Chile/risotto/workflows/CI/badge.svg)
[![Inria](https://img.shields.io/badge/Made%20in-Inria-%23e63312)](http://inria.cl)
[![License: CeCILLv2.1](https://img.shields.io/badge/license-CeCILL--v2.1-orange)](https://cecill.info/licences.en.html)

This repository contains the source files that support the paper:

* Rodolfo Palma, Luis Martí and Nayat Sánchez-Pi (2020) *Predicting Accidents in the Mining Industry with a Multi-Task Learning Approach*. submitted to the [Thirty-Third Annual Conference on Innovative Applications of Artificial Intelligence (IAAI-21)](https://aaai.org/Conferences/AAAI-21/iaai-21-call/).

## Abstract

The mining sector is a very relevant part of the Chilean economy, representing more than 14% of the country’s GDP and more than 50% of its exports. However, mining is also a high-risk activity where health, safety, and environmental aspects are fundamental concerns to take into account to render it viable in the longer term. The Chilean National Geology and Mining Service (SERNAGEOMIN, after its name in Spanish) is in charge of ensuring the safe operation of mines. On-site inspections are their main tool in order to detect issues, propose corrective measures, and track the compliance of those measures.  Consequently, it is necessary to create inspection programs relying on a data-based decision-making strategy.

This paper reports the work carried out in one of the most relevant dimensions of said strategy: predicting the mining worksites accident risk. That is, how likely it is a mining worksite to have accidents in the future. This risk is then used to create a priority ranking that is used to devise the inspection program. Estimating this risk at the government regulator level is particularly challenging as there is a very limited and biased data.

Our main contribution is to apply a multi-task learning approach to train the risk prediction model in such a way that is able to overcome the constraints of the limited availability of data by fusing different sources. As part of this work, we also implemented a human-experience-based model that captures the procedures currently used by the current experts in charge of elaborating the inspection priority ranking.

The mining worksites risk rankings built by model achieve a 121.2% NDCG performance improvement over the rankings based on the currently used experts’ model and outperforms the non-multi-task learning alternatives.

## Installing

To install the project dependencies run:

```zsh
pip install -r requirements.txt
```

## Experiments

We provide scripts to replicate the results reported in our paper.
The baseline results based on tree models can be obtained by running the [`BaselineModels.ipynb`](https://github.com/Inria-Chile/mining-risk-multitasking-model/blob/master/BaselineModels.ipynb) notebook in the root of the current repository.

The results of the single-task and multitask neural networks can be replicated by running the scripts described in the following table.

| Script name          | Model           | Description                                                  |
|----------------------|-----------------|--------------------------------------------------------------|
| [train_classification.sh](https://github.com/Inria-Chile/mining-risk-multitasking-model/blob/master/experiments/train_classification.sh) | NN classifier   | Neural network trained in a single classification task setup |
| [train_regression.sh](https://github.com/Inria-Chile/mining-risk-multitasking-model/blob/master/experiments/train_regression.sh) | NN regressor    | Neural network trained in a single regressionn task setup    |
| [train_multitask.sh](https://github.com/Inria-Chile/mining-risk-multitasking-model/blob/master/experiments/train_multitask.sh) | Multitask model | Neural network trained in a multitask setup                  |

The Python script `run.py` should be run in order to train the neural network models with custom parameters.
The script receives the following command line flags.

```zsh
$ python run.py --help
usage: run.py [-h] [--logger [LOGGER]] [--checkpoint_callback [CHECKPOINT_CALLBACK]] [--early_stop_callback [EARLY_STOP_CALLBACK]] [--default_root_dir DEFAULT_ROOT_DIR]
              [--gradient_clip_val GRADIENT_CLIP_VAL] [--process_position PROCESS_POSITION] [--num_nodes NUM_NODES] [--num_processes NUM_PROCESSES] [--gpus GPUS]
              [--auto_select_gpus [AUTO_SELECT_GPUS]] [--tpu_cores TPU_CORES] [--log_gpu_memory LOG_GPU_MEMORY] [--progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE]
              [--overfit_batches OVERFIT_BATCHES] [--track_grad_norm TRACK_GRAD_NORM] [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--fast_dev_run [FAST_DEV_RUN]]
              [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--max_epochs MAX_EPOCHS] [--min_epochs MIN_EPOCHS] [--max_steps MAX_STEPS] [--min_steps MIN_STEPS]
              [--limit_train_batches LIMIT_TRAIN_BATCHES] [--limit_val_batches LIMIT_VAL_BATCHES] [--limit_test_batches LIMIT_TEST_BATCHES]
              [--val_check_interval VAL_CHECK_INTERVAL] [--log_save_interval LOG_SAVE_INTERVAL] [--row_log_interval ROW_LOG_INTERVAL] [--distributed_backend DISTRIBUTED_BACKEND]
              [--precision PRECISION] [--print_nan_grads [PRINT_NAN_GRADS]] [--weights_summary WEIGHTS_SUMMARY] [--weights_save_path WEIGHTS_SAVE_PATH]
              [--num_sanity_val_steps NUM_SANITY_VAL_STEPS] [--truncated_bptt_steps TRUNCATED_BPTT_STEPS] [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--profiler [PROFILER]]
              [--benchmark [BENCHMARK]] [--deterministic [DETERMINISTIC]] [--reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]] [--auto_lr_find [AUTO_LR_FIND]]
              [--replace_sampler_ddp [REPLACE_SAMPLER_DDP]] [--terminate_on_nan [TERMINATE_ON_NAN]] [--auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]]
              [--prepare_data_per_node [PREPARE_DATA_PER_NODE]] [--amp_level AMP_LEVEL] [--val_percent_check VAL_PERCENT_CHECK] [--test_percent_check TEST_PERCENT_CHECK]
              [--train_percent_check TRAIN_PERCENT_CHECK] [--overfit_pct OVERFIT_PCT] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--dataset_path DATASET_PATH]
              [--wandb_logging WANDB_LOGGING] [--wandb_name WANDB_NAME] [--dataloader_workers DATALOADER_WORKERS] [--root_dir ROOT_DIR]
              [--fill_missing_regression FILL_MISSING_REGRESSION] [--regression_task REGRESSION_TASK] [--classification_task CLASSIFICATION_TASK] [--learning_rate LEARNING_RATE]
              [--classifier_lambda CLASSIFIER_LAMBDA] [--input_size INPUT_SIZE] [--hidden_size HIDDEN_SIZE] [--tanh_loss TANH_LOSS] [--regressor_activation REGRESSOR_ACTIVATION]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --dataset_path DATASET_PATH
  --wandb_logging WANDB_LOGGING
  --wandb_name WANDB_NAME
  --dataloader_workers DATALOADER_WORKERS
  --root_dir ROOT_DIR
  --fill_missing_regression FILL_MISSING_REGRESSION
  --regression_task REGRESSION_TASK
  --classification_task CLASSIFICATION_TASK
  --learning_rate LEARNING_RATE
  --classifier_lambda CLASSIFIER_LAMBDA
  --input_size INPUT_SIZE
  --hidden_size HIDDEN_SIZE
  --tanh_loss TANH_LOSS
  --regressor_activation REGRESSOR_ACTIVATION
```

## Citing

```bibtex
@techreport{palma2020:mining-accident-risk,
    author = {Palma, Rodolfo and Mart{\'{\i}}, Luis and Sanchez-Pi, Nayat}
    title = {Predicting Accidents in the Mining Industry with a Multi-Task Learning Approach},
    year = {2020},
    institution = {Inria Research Center in Chile},
    note = {submitted to the Thirty-Third Annual Conference on Innovative Applications of Artificial Intelligence (IAAI-21)}
}
```
