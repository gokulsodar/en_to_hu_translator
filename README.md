# English to Hungarian Translation with T5-Small

This repository contains code for fine-tuning a T5-small model on English to Hungarian translation tasks using the OPUS Books dataset.

## Model

The fine-tuned model is too large to be stored on GitHub, so it is hosted on Hugging Face:

[https://huggingface.co/anonymus16/en-to-hu_finetuned-t5-small] <!-- Add your Hugging Face link here -->

## Dataset & Training

Training was performed on an NVIDIA P100 GPU via Kaggle:

[https://www.kaggle.com/code/gokulsodar/t5-eng-to-hu] <!-- Add your Kaggle notebook link here -->

## Training Details

- Base model: t5-small
- Dataset: OPUS Books English-Hungarian parallel corpus
- Mixed precision (FP16) training for optimized performance
- Hyperparameter optimization with Optuna
- Training progress monitored with TensorBoard

## Libraries Used

- PyTorch
- Transformers
- Datasets
- Optuna
- NumPy
- Scikit-learn

## Project Structure

- `en_hu_translation.ipynb`: Main notebook containing all training code
- Training leverages HuggingFace's Trainer API with cosine learning rate scheduler with restarts
- TensorBoard logging to monitor training metrics

## Usage

1. Install dependencies:
```
pip install torch transformers datasets optuna numpy scikit-learn
```

2. Run the notebook to:
   - Load and preprocess the English-Hungarian dataset
   - Optimize hyperparameters with Optuna
   - Train final model with optimized settings
   - Evaluate and test translations

## Local Testing

If you want to use the pre-trained model locally:

1. Download the model files from Hugging Face (excluding the README.md) and store them in a folder called `t5-small-hungarian-translator`
2. Install the necessary libraries:
```
pip install torch transformers
```
3. Use the model with the provided `testing.py` script (or create one using the example below)

## testing.py

```python
from transformers import AutoTokenizer, T5ForConditionalGeneration

model_path = "t5-small-hungarian-translator"
loaded_model = T5ForConditionalGeneration.from_pretrained(model_path)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)

# Then use it for translation
def translate(text):
    input_text = "translate English to Hungarian: " + text
    inputs = loaded_tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    outputs = loaded_model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    return loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test translation with a sample sentence
text = input()
translated_text = translate(text)    

print(f"English: {sample_text}")
print(f"Hungarian: {translated_text}")
```
