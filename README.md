# Arabic Dialect Classification: A Comparative Study of TF-IDF and MARBERT

**Exploring traditional machine learning and contextual language modeling for Arabic dialect recognition.**

## About the Project

Arabic dialect recognition is challenging because regional varieties often share vocabulary while differing in spelling, expressions, and linguistic structure.

This project investigates two approaches to this problem: a feature-based classifier using TF-IDF and Logistic Regression, and a transformer-based classifier built on MARBERT. The goal is to examine their performance on a multi-class Arabic dialect classification task.

## The Experiment

The workflow uses Arabic text as input and the corresponding country label as the target.

The dataset contains **112,000 records** across **16 dialect categories**, including Saudi, Egyptian, Iraqi, Moroccan, and other regional varieties, as well as Modern Standard Arabic.

Before training, the data was explored and cleaned. Missing text and labels were removed, and Arabic text normalization was applied while preserving dialect-specific spelling patterns.

The data was split into training and validation sets using an 85:15 stratified split.

## Two Modeling Approaches

### Baseline: TF-IDF + Logistic Regression

The baseline converts text into numerical features using TF-IDF with unigram and bigram representations. The vocabulary is limited to 50,000 features, and Logistic Regression is trained with class balancing.

This approach provides a conventional, computationally efficient reference point for the experiment.

### Contextual Model: MARBERT

MARBERT is fine-tuned to classify Arabic dialects using contextual representations of the input text.

The documented training setup includes a learning rate of `1e-5`, a batch size of `16`, gradient accumulation over two steps, two training epochs, a maximum sequence length of `64` tokens, and FP16 mixed-precision training.

## Evaluation and Findings

The models were compared using Accuracy, Macro F1-score, and confusion matrices.

| Evaluation metric | TF-IDF + Logistic Regression | MARBERT |
|---|---:|---:|
| Accuracy | 72% | 79% |
| Macro F1-score | 60% | 67% |

MARBERT achieved higher reported scores on both metrics, with a difference of **7 percentage points** in Accuracy and Macro F1-score.

The confusion-matrix analysis also examined classification errors between related dialects, including Gulf dialects and North African varieties.

These findings suggest that contextual representations were more effective than the TF-IDF baseline in this experiment. The results should be interpreted in the context of the dataset distribution and evaluation setup.

## From Model to Application

Beyond model evaluation, the project includes a Gradio-based interface for interactive inference.

Users can enter an Arabic sentence and receive a predicted dialect with a confidence score, providing a practical way to explore the trained model's behavior.

## Technical Stack

`Python` · `Pandas` · `Scikit-learn` · `TF-IDF` · `Logistic Regression` · `MARBERT` · `Hugging Face Transformers` · `Gradio` · `Google Colab`

## Project Resources

- **Implementation:** [Open the Google Colab notebook](https://colab.research.google.com/drive/17AFzrdWwsLayfIL_rEz0XM03upbiW8Lg?usp=sharing)
- **Project report:** `Arabic_Dialect_Identification_Report.pdf`
- **Notebook:** `Arabic_Dialect_Identification.ipynb`

Confirm that the report and notebook filenames match the files available in this repository.

## What Could Be Improved Next?

- Evaluate the models on an independent test set.
- Report precision, recall, and F1-score for each dialect.
- Investigate errors between linguistically similar dialects.
- Assess performance on underrepresented classes.
- Document deployment and reproducibility requirements.

## Academic Background

Applied Natural Language Processing  
Umm Al-Qura University
