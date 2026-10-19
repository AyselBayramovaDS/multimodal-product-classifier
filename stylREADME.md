# Multi-Modal Product Classifier (Image + Text)

Classifies fashion products into 37 `subCategory` classes using the product image, the product text, and both, and compares the three models under identical conditions.

Dataset: [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) (Kaggle).

## Files
- `fashionpi.ipynb`: full pipeline with outputs (data audit, split, pipelines, three models, evaluation, error analysis, verdict, all plots)
- `note.md`: written report (class distribution, split, fusion design, freezing strategy, comparison table, error analysis, verdict)

## Models (same split, 3 epochs, AdamW, CrossEntropy, macro-F1 model selection)
- **A:** ResNet18 (ImageNet) + `Linear(512, 37)`, image only
- **B:** DistilBERT (`distilbert-base-uncased`) + `Linear(768, 37)`, text only
- **C:** ResNet18 + DistilBERT, features concatenated (1280) then an MLP (late fusion)

## Result (test set, macro F1)
Image 0.767 | Text **0.914** | Fusion 0.900. Fusion did not beat text-only (difference not significant); text is the dominant modality.

## How to run
1. Open the notebook on Kaggle with the dataset added as input, GPU enabled and Internet on (needed to download the DistilBERT tokenizer/weights).
2. Check `base` in the path cell (`/kaggle/input/datasets/paramaggarwal/fashion-product-images-small`) matches your input.
3. Run all cells top to bottom (about 25–30 minutes on a Kaggle GPU).

Requirements: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`, `torchvision`, `transformers`, `Pillow`.

Notes: results are from a single training run (no fixed torch seed), so exact numbers can vary slightly between runs. The split uses `random_state=42`.