# Akkadian Cuneiform — Machine Translation

Machine translation project for the **Deep Past Initiative** Kaggle challenge, translating Akkadian transliterations into English.

The project explores a complete machine translation workflow, from data exploration and baseline evaluation to **NLLB-200 inference and fine-tuning**.

## Approach

* **Data exploration:** inspect the competition dataset, source/target fields, sequence lengths, and missing data.
* **Baseline:** establish a simple translation baseline and generate competition-format submissions.
* **NLLB-200:** use `facebook/nllb-200-distilled-600M` as a multilingual sequence-to-sequence translation model.
* **Fine-tuning:** fine-tune NLLB on the competition training data using Hugging Face Transformers and PyTorch.
* **Evaluation:** evaluate translations locally using the competition's combined **BLEU + chrF++** metric.
* **Kaggle submission:** run inference on the competition test set and generate the required `submission.csv`.

Because Akkadian is not directly supported by NLLB, the project uses the NLLB Arabic language code (`arb_Arab`) as a proxy for the source language.

## Repository Structure

```text
├── config.py
├── download_data.py
├── explore_data.py
├── evaluate.py
├── run_baseline.py
├── run_local_eval.py
├── save_nllb_for_kaggle.py
├── zip_nllb_for_kaggle.py
├── notebooks/
│   ├── finetune_nllb.ipynb
│   └── submission.ipynb
└── requirements.txt
```

### Key notebooks

**`notebooks/finetune_nllb.ipynb`**
Kaggle GPU notebook for fine-tuning NLLB-200 on the competition training data.

**`notebooks/submission.ipynb`**
Kaggle submission notebook that loads an NLLB model, translates the test set, and creates the required `submission.csv`.

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* NLLB-200
* Pandas
* Kaggle
* BLEU / chrF++

## Running the Project

Install the dependencies:

```bash
python -m venv venv
```

Activate the environment and install:

```bash
pip install -r requirements.txt
```

Competition data can be downloaded using:

```bash
python download_data.py
```

For local evaluation:

```bash
python run_local_eval.py --n 500
```

The NLLB fine-tuning and competition submission notebooks are designed to run on Kaggle.

## Competition

**Deep Past Initiative — Machine Translation**

Task: **Akkadian transliteration → English**

The competition evaluates translation quality using a combination of BLEU and chrF++.
