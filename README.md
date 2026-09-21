# Deep Past Initiative — Machine Translation (Kaggle)

This repo is a **step-by-step pipeline** to participate in the [Deep Past Challenge](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/overview): **translate Akkadian (cuneiform) to English**. It provides scripts for downloading data, exploring it, running a baseline, and evaluating with the competition metric locally. All logic is commented so you can follow and extend it from the start.

**Competition goal:** Build MT systems to decode ancient Mesopotamian trade records (contracts, letters, loans, receipts) from the early 2nd millennium BCE.

**Evaluation:** Score = √(BLEU × chrF++) — both metrics must be good.

---

## Step 1 — Kaggle account and API

1. **Sign up / log in** at [kaggle.com](https://www.kaggle.com).
2. **Join the competition:**  
   [Deep Past Initiative - Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation) → **“Join Competition”** (accept rules).
3. **Create API credentials:**  
   Profile (top right) → **Settings** → **API** → **Create New Token**.  
   - **If a file downloads:** you get `kaggle.json`. Put it in `C:\Users\<YourUsername>\.kaggle\` (Windows) or `~/.kaggle/` (Mac/Linux). On Mac/Linux run `chmod 600 ~/.kaggle/kaggle.json`.
   - **If only a token is shown (no file):** use the environment variable instead (see below).
4. **Install Kaggle CLI:** `pip install kaggle`

**Using the token without a file (env var):**  
If Kaggle only gave you a token to copy (no `kaggle.json` download), set it in your shell before running any `kaggle` command or `python download_data.py`:

- **Windows (PowerShell):** `$env:KAGGLE_API_TOKEN = "your_token_here"`
- **Windows (CMD):** `set KAGGLE_API_TOKEN=your_token_here`
- **Mac/Linux:** `export KAGGLE_API_TOKEN=your_token_here`

Check that it works: `kaggle competitions list` (you should see the competition list).

---

## Step 2 — Project setup

```bash
cd kaggle_deep_past
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

---

## Step 3 — Download competition data

From the project folder (with venv active):

```bash
python download_data.py
```

Or manually:

```bash
kaggle competitions download -c deep-past-initiative-machine-translation -p data/raw
```

Then extract the zip. **Windows (PowerShell):** from `data/raw`, run  
`Expand-Archive -Path .\*.zip -DestinationPath . -Force`  
**Mac/Linux:** `cd data/raw && unzip *.zip`

**Check the Data tab on Kaggle** for exact file names. Typical layout:
- **Train:** e.g. `train.csv` or `train.tsv` — columns for Akkadian (source) and English (target).
- **Test:** e.g. `test.csv` — IDs and Akkadian source only.
- **Sample submission:** e.g. `sample_submission.csv` — `id` and `translation` (or `target`).

---

## Step 4 — Explore the data

```bash
python explore_data.py
```

This will:
- Load train/test (and sample submission if present).
- Print shapes, column names, and sample rows.
- Show basic stats (lengths, missing values).

Use this to confirm column names for the rest of the pipeline (e.g. `translation`, `Akkadian`, `English`).

---

## Step 5 — Evaluate locally (BLEU + chrF++)

The competition score is **sqrt(BLEU × chrF++)** (both normalized 0–1).

**Quick local eval (train sample as validation):**

```bash
python run_local_eval.py --n 500
```

This uses the first 500 rows of train: references = gold `translation`, predictions = current translator (dummy baseline or your model) on `transliteration`, writes `data/processed/refs.txt` and `preds_baseline.txt`, then runs the metric. Use it to check your model before submitting.

**Manual eval (your own refs/preds files):**

```bash
python evaluate.py --references path/to/references.txt --predictions path/to/predictions.txt
```

One segment per line in each file.

---

## Step 6 — Baseline and submission

1. **Baseline:**  
   Run the placeholder baseline to get a submission file and confirm the pipeline:

   ```bash
   python run_baseline.py
   ```

   Output: `data/submissions/submission_baseline.csv`. Replace `dummy_translate()` in `run_baseline.py` with a real model to improve (e.g. fine-tuned NLLB/mBART).

2. **Submission file:**  
   Must match the format on the **Evaluation** tab (usually CSV with `id` and one translation column).  
   Example:
   ```text
   id,translation
   0,"Your English translation here."
   1,"Another translation."
   ```

3. **Submit on Kaggle (this competition = notebook re-run):**  
   Kaggle **re-runs your notebook** with a hidden test set and uses the **output file** you choose as your submission. Use the submission notebook:

   - **Notebook:** `notebooks/submission.ipynb` (in this repo). Copy it to Kaggle: **Code** → **New Notebook** → paste the cells, or upload the file.
   - In the notebook: data is read from `/kaggle/input/deep-past-initiative-machine-translation/`; the notebook writes **submission.csv** to `/kaggle/working/`.
   - After **Run All**: in the right panel, **Output** → **+ Add output** → select **submission.csv**. Then **Save Version** → **Save & Run All (Commit)**. When the version finishes, open it → **Submit to Competition** and choose this notebook’s output.

Check the competition’s **Submission** / **Evaluation** section for the exact output file name and format.

---

## Step 7 — NLLB (no fine-tune) and other data

**Use NLLB without fine-tuning**

1. **Save the model for Kaggle (run once, with internet):**
   ```bash
   pip install torch transformers accelerate
   python save_nllb_for_kaggle.py
   ```
   This writes the NLLB-200 distilled 600M model to `data/models/nllb-200-distilled-600M/`.

2. **Create a Kaggle Dataset from the model folder:**
   - **Zip the folder** (the model is ~2.3 GB; PowerShell’s Compress-Archive often fails with “Stream was too long”, so use one of these):
     - **Recommended (Windows):** From the project root:
       ```bash
       python zip_nllb_for_kaggle.py
       ```
       This creates `data/models/nllb-200-distilled-600M.zip` (takes a few minutes).
     - **Windows (Explorer):** Right‑click `data\models\nllb-200-distilled-600M` → **Send to** → **Compressed (zipped) folder** (may fail on very large size).
     - **Mac/Linux:** `cd data/models && zip -r nllb-200-distilled-600M.zip nllb-200-distilled-600M`
   - Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**.
   - **Upload** the zip (drag & drop or click to select `nllb-200-distilled-600M.zip`).
   - **Title:** e.g. `nllb-200-distilled-600M` (or any name you like).
   - **Description:** optional, e.g. "NLLB-200 distilled 600M for offline use in Deep Past notebook."
   - Click **Create**. Your dataset will have a URL like `kaggle.com/datasets/yourusername/nllb-200-distilled-600M`.

3. **Add the dataset to your submission notebook:**
   - Open your **submission notebook** on Kaggle (the one you submit to the competition).
   - In the right panel, under **Input**, click **+ Add input** (or **Add data**).
   - Search for your dataset by name (e.g. `nllb-200-distilled-600M`) or open it from "Your work" / "My Datasets", then **Add**.
   - The notebook will see it at `/kaggle/input/<your-dataset-slug>/`. The notebook code already looks for any input that contains `config.json` and uses it as the model. **Save** and run with **Internet OFF**; the model loads from this input.

4. **Local run with NLLB:**
   ```bash
   python run_baseline.py --model      # submission CSV using NLLB
   python run_local_eval.py --model --n 500   # local BLEU/chrF++ with NLLB
   ```

**Other competition data:** The challenge zip includes more than `train.csv` and `test.csv` (e.g. `published_texts.csv`, lexicons, bibliography). Check the **Data** tab on Kaggle for descriptions; you can use them for preprocessing, vocabulary, or future fine-tuning.

**Kaggle suggestions:** See the competition **Overview**, **Code** (public notebooks), and **Discussion** for suggested approaches and shared tips.

---

## Free GPU and fine-tuning

**You get free GPU on Kaggle.** In any notebook: right panel → **Settings** (or **Accelerator**) → **GPU T4 x2** (or **P100**). You have a weekly quota (e.g. 30 h GPU); sessions can be limited to 9–12 h. No payment needed. The **submission** notebook must run with **Internet OFF**; the **training** notebook can use **Internet ON** and GPU.

**Other free GPU options:**  
- **Google Colab:** free tier includes GPU (T4), with session and weekly limits.  
- **Kaggle** is often the simplest here because the competition data and your NLLB dataset are already there.

**Fine-tune NLLB on the competition train set (Kaggle GPU):**

1. **Training notebook:** Use `notebooks/finetune_nllb.ipynb`. Copy it to Kaggle (or create a new notebook and paste the cells).
2. **Enable GPU:** In the notebook, **Settings** → **Accelerator** → **GPU**. **Add inputs:** In the right panel click **+ Add input** (or **Add data**). Add the **competition** dataset (search e.g. "deep past initiative"); if you created the notebook from the competition page it may already be there. Optionally add **your NLLB dataset** (search its name or use "Your work" → "My Datasets") so training starts from your saved model; otherwise the notebook downloads base NLLB from Hugging Face.
3. **Run All.** The notebook loads `train.csv`, fine-tunes NLLB with `Seq2SeqTrainer`, and saves the model to `/kaggle/working/finetuned_nllb/`.
4. **Use the fine-tuned model in submission:** After the run, open **Output** → download the `finetuned_nllb` folder (or use “Save Version” and create a dataset from the output). Zip it, create a **new Kaggle Dataset**, then add that dataset as an **Input** to your **submission notebook**. The submission notebook will load the fine-tuned model the same way it loads the base NLLB (it searches for a dir containing `config.json`).

You can change `MAX_SAMPLES` and `EPOCHS` in the training notebook (e.g. full data and 3–5 epochs for a stronger model; start with 2000 samples and 2 epochs for a quick test).

---

## Project layout

```text
kaggle_deep_past/
├── README.md              # This guide
├── requirements.txt
├── .gitignore
├── config.py              # Paths and options
├── download_data.py       # Download competition data
├── explore_data.py        # Load and inspect data
├── evaluate.py            # BLEU, chrF++, combined score
├── run_local_eval.py      # Local eval on train sample (refs + preds → evaluate.py)
├── run_baseline.py        # Baseline or NLLB (--model) → submission CSV
├── save_nllb_for_kaggle.py  # Save NLLB model for Kaggle Dataset (no internet)
├── zip_nllb_for_kaggle.py   # Zip model folder (use this; PowerShell fails on large dirs)
├── data/
│   ├── raw/               # Downloaded zips and extracted files
│   ├── models/            # Local NLLB cache (from save_nllb_for_kaggle.py)
│   ├── processed/         # Processed train/val/test (optional)
│   └── submissions/       # Prediction CSVs for upload
└── notebooks/
    ├── submission.ipynb   # Notebook for Kaggle submission (re-run with hidden test)
    └── finetune_nllb.ipynb  # Fine-tune NLLB on train set (run on Kaggle with GPU)
```

---

## Quick reference

| What              | Where |
|-------------------|--------|
| Competition page  | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation |
| Metric            | √(BLEU × chrF++) |
| Task              | Akkadian (cuneiform) → English |
| Data / submission | See **Data** and **Evaluation** tabs on Kaggle |

Good luck — and have fun unearthing ancient voices.
