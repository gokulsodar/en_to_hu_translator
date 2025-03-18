# English to Hungarian Translation with T5-Small

This repository contains code for fine-tuning a T5-small model on English to Hungarian translation tasks using the OPUS Books dataset.

## Model

The fine-tuned model is too large to be stored on GitHub, so it is hosted on Hugging Face:

[] <!-- Add your Hugging Face link here -->

## Dataset & Training

Training was performed on an NVIDIA P100 GPU via Kaggle:

[https://www.kaggle.com/code/gokulsodar/t5-eng-to-hu] <!-- Add your Kaggle notebook link here -->

## Training Details

The original T5-small model only supports English to French, English to German, and English to Romanian translations. I fine-tuned it to add English to Hungarian translation as well.

- Base model: t5-small
- Dataset: OPUS Books English-Hungarian parallel corpus
- Mixed precision (FP16) training for optimized performance
- Hyperparameter optimization with Optuna
- Training progress monitored with TensorBoard

## Libraries Used

- PyTorch
- Transformers
- Optuna

## Project Structure

- `t5-eng-to-hu.ipynb`: Jupyter notebook containing the outputs of the training process
- TensorBoard logging to monitor training metrics

## Deployment

The model was deployed in GCP Cloud Run as a Docker container:

[https://translatorimg-621297051030.asia-south1.run.app] <!-- Add your GCP Cloud Run link here -->

