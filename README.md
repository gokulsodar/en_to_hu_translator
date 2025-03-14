# English to Hungarian T5-small Translation Model

This project fine-tunes the T5-small model on English-Hungarian translation data using the `opus_books` dataset. It includes hyperparameter optimization with Optuna and provides a streamlined workflow for training and evaluating neural machine translation models.

## Features

- Fine-tuning T5-small for English to Hungarian translation
- Hyperparameter optimization using Optuna
- TensorBoard integration for training visualization
- Mixed precision training support for GPU acceleration
- Complete pipeline from data preprocessing to model evaluation

## Requirements

```
torch
transformers
datasets
optuna
numpy
sklearn
tensorboard
```

## Dataset

This project uses the `opus_books` dataset (English-Hungarian split) from Hugging Face's datasets library. The dataset consists of translations from books and literature, providing a diverse vocabulary for training.

## Project Structure

- **Data Loading**: Load and preprocess the English-Hungarian translation dataset
- **Tokenization**: Prepare the data using T5's tokenizer with task-specific prefixes
- **Hyperparameter Optimization**: Use Optuna to find optimal training parameters
- **Model Training**: Train the T5 model using the best hyperparameters
- **Evaluation**: Test the model with sample translations

## Usage

### 1. Setup Environment

Create a Python environment with the required packages:

```bash
conda create -n translation-env python=3.10
conda activate translation-env
pip install torch transformers datasets optuna scikit-learn tensorboard
```

### 2. Run the Notebook

The entire process is contained in the Jupyter notebook. Each cell is clearly labeled with its purpose:

- `import-libraries`: Load necessary Python libraries
- `load-dataset`: Import the English-Hungarian dataset
- `preprocess-data`: Tokenize and prepare data for training
- `optuna-objective`: Define the hyperparameter search space
- `run-optuna`: Execute hyperparameter optimization
- `final-training`: Train the model with optimal parameters
- `evaluate-model`: Test the model with example translations

### 3. Custom Translation

After training, you can use the `translate_text()` function to translate your own text:

```python
sample_text = "I would like to visit Budapest someday."
translated_text = translate_text(sample_text)
print(f"English: {sample_text}")
print(f"Hungarian: {translated_text}")
```

## Hyperparameters

The optimization process tunes the following hyperparameters:

- Learning rate
- Warmup steps
- Gradient accumulation steps
- Weight decay
- Batch size

## Performance Monitoring

Training progress can be monitored through TensorBoard:

```bash
tensorboard --logdir=./t5-small-translation-tensorboard
```

## Model Saving & Loading

The final model is saved to the `./t5-small-translation-final` directory and can be loaded for inference:

```python
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("./t5-small-translation-final")
model = T5ForConditionalGeneration.from_pretrained("./t5-small-translation-final")
```

## Future Improvements

- Increase training data with additional Hungarian-English datasets
- Experiment with larger T5 variants (T5-base, T5-large)
- Implement more sophisticated evaluation metrics (BLEU, ROUGE)
- Add model quantization for faster inference
- Create a simple web interface for demo purposes

## License

[MIT License]

## Acknowledgments

- Hugging Face for the Transformers library and datasets
- OPUS project for the translation data
