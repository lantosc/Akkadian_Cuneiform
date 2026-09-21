"""
evaluate.py — Step 5: Local evaluation with BLEU, chrF++, and competition score.

The Deep Past competition uses the geometric mean of BLEU and chrF++ (normalized 0–1):
  score = √(BLEU × chrF++)

This script reads two text files (one segment per line), computes BLEU and chrF++
with sacrebleu, then prints the combined score. Use it to evaluate your model on a
validation set (e.g. references = gold English, predictions = model output).

Usage:
  python evaluate.py --references refs.txt --predictions preds.txt
  python evaluate.py -r path/to/references.txt -p path/to/predictions.txt

Requirements: pip install sacrebleu (see requirements.txt).
"""

import argparse
import math
import sys


def main():
    parser = argparse.ArgumentParser(description="Compute BLEU, chrF++, and competition score")
    parser.add_argument("--references", "-r", required=True, help="Reference translations (one per line)")
    parser.add_argument("--predictions", "-p", required=True, help="Model predictions (one per line)")
    args = parser.parse_args()

    try:
        import sacrebleu
    except ImportError:
        print("Install sacrebleu: pip install sacrebleu")
        sys.exit(1)

    with open(args.references, "r", encoding="utf-8") as f:
        refs = [line.strip() for line in f]
    with open(args.predictions, "r", encoding="utf-8") as f:
        preds = [line.strip() for line in f]

    if len(refs) != len(preds):
        print(f"Length mismatch: {len(refs)} references vs {len(preds)} predictions")
        sys.exit(1)

    # SacreBLEU expects refs as list of lists (one ref per segment; multiple refs per segment allowed)
    refs_list = [[r] for r in refs]

    bleu = sacrebleu.corpus_bleu(preds, refs_list)
    chrf = sacrebleu.corpus_chrf(preds, refs_list)

    # Normalize to 0–1 (sacrebleu reports 0–100)
    bleu_score = bleu.score / 100.0
    chrf_score = chrf.score / 100.0

    # Competition metric: geometric mean of BLEU and chrF++ (both in 0–1)
    combined = math.sqrt(bleu_score * chrf_score) if (bleu_score > 0 and chrf_score > 0) else 0.0

    print(f"BLEU:     {bleu_score:.4f} (sacrebleu: {bleu.score:.2f})")
    print(f"chrF++:   {chrf_score:.4f} (sacrebleu: {chrf.score:.2f})")
    print(f"Score sqrt(BLEU*chrF++): {combined:.4f}")


if __name__ == "__main__":
    main()
