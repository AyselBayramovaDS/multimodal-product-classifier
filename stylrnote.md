# Multi-Modal Product Classifier (Image + Text): note.md

**Dataset:** [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) (Kaggle, `paramaggarwal`)
**Environment:** Kaggle Notebook with GPU (CUDA), full dataset, no subsampling
**Target:** `subCategory` (37 classes after cleaning)
**Notebook:** `fashionpi.ipynb` (all numbers below come from its outputs; all plots are in the notebook)

---

## Summary

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| A: Image-only (ResNet18) | 0.9618 | 0.7670 | 0.9597 |
| B: Text-only (DistilBERT) | 0.9890 | **0.9142** | 0.9882 |
| C: Fused (late fusion) | 0.9878 | 0.8998 | 0.9870 |

**Verdict: fusion did not help.** It is far better than image-only (+0.133 macro F1) but 0.014 below text-only, and that gap is not statistically distinguishable from zero (bootstrap 95% CI [−0.028, +0.010]). Product titles already name the category, so text is the dominant modality in this dataset.

---

## 1. Data audit

| Step | Rows affected | Rows remaining |
|---|---|---|
| Lines in `styles.csv` (excluding header) | | 44,446 |
| Malformed rows skipped by `on_bad_lines="skip"` | **22** | 44,424 |
| Missing `productDisplayName` (dropped) | **7** | 44,417 |
| No matching image file (dropped) | **5** | 44,412 |
| Classes with < 10 samples (dropped, 8 classes) | **32** | **44,380** |

- Image folder contained 44,441 files; `id` is unique (44,424 unique ids after reading).
- `usage` (317 missing) and `baseColour` (15 missing) were **not** dropped; they were filled with empty strings so no rows were lost. `season` and `year` are not used.
- Metadata sizes: `masterCategory` 7, `subCategory` 45, `articleType` 143 distinct values.

### Category level
`subCategory` was chosen: `masterCategory` (7 classes) is too easy, and `articleType` (143 classes) would produce classes with only a handful of examples. `subCategory` (45 classes) is learnable and keeps both modalities meaningful.

### Class distribution (before removing rare classes; bar chart in the notebook)
The data is strongly imbalanced. The largest class, **Topwear, has 15,398 of 44,412 rows (34.7%)**, followed by Shoes (7,343) and Bags (3,053), while the smallest classes have 1–7 rows (Home Furnishing 1, Vouchers 1, Sports Accessories 3, Beauty Accessories 4, Wristbands 4, Umbrellas 6, Perfumes 6, Water Bottle 7).

### Rare-class decision: drop (< 10 samples)
Eight classes with fewer than 10 samples were dropped (32 rows, about 0.07% of the data), leaving 37 classes and 44,380 rows.

- With a stratified 70/15/15 split such classes would have 0–1 samples in validation/test; `stratify` fails for classes with a single sample.
- Per-class F1 on 0–1 test samples is 0 or 1 by chance, and since macro F1 weights every class equally, this noise would make model comparisons unreliable.
- Merging them into an "Other" class was rejected because it would mix unrelated products (e.g. umbrellas, vouchers, water bottles) into a meaningless category.
- The threshold of 10 is my own choice, not given by the task.

### Why macro F1 is the primary metric
Topwear alone is 34.7% of the rows, so a model that always predicts "Topwear" would already reach about 35% accuracy, and large classes dominate accuracy in general. Macro F1 averages the per-class F1 with equal weight, so rare classes count as much as frequent ones. The results show the effect clearly: the image model is only 2.7 accuracy points behind the text model, but 14.7 macro-F1 points behind.

---

## 2. Split strategy

- One stratified split on the integer label (`LabelEncoder`, 0–36): **70% train / 15% validation / 15% test = 31,066 / 6,657 / 6,657 rows**, `random_state=42`.
- Done in two steps (70 / 30, then the 30% halved), both with `stratify`.
- The same `train_df / val_df / test_df` are used by all three models. The splits were also saved to `train.csv`, `val.csv`, `test.csv`.
- Check: the top class (Topwear) is 34.7% in each of the three parts; the smallest class has 2 samples in the test set.
- Validation is used only to select the best epoch; the test set is used once per model, at the end.

---

## 3. Input pipelines

**Text.** `productDisplayName + " " + baseColour + " " + usage`, tokenised with the pretrained `distilbert-base-uncased` tokenizer, padded/truncated to 32 tokens (with attention mask). `masterCategory` and `articleType` were deliberately **excluded**: they are too close to the target `subCategory` and would leak the label. `gender`, `season`, `year` were left out as carrying little category information.

**Images.** Resized to 224×224 and normalised with ImageNet mean/std. Augmentation on the **training set only**: random horizontal flip, rotation up to 10°, colour jitter (brightness/contrast 0.2). Validation and test use only resize + normalisation. The source images are low resolution (60×80 according to the dataset description) and are upsampled to 224.

**Shared Dataset.** A single `ProductDataset` returns `image`, `input_ids`, `attention_mask`, `label` for each sample; all models use the same dataset class and DataLoaders (batch size 64; shuffle only for train). A sanity batch gave shapes `[64, 3, 224, 224]`, `[64, 32]`, `[64]`.

---

## 4. Models and training conditions

All three models are trained and evaluated by the same `fit()` function: same split, **3 epochs**, **AdamW**, **CrossEntropyLoss**, batch size 64, the same metric (macro F1 on validation after each epoch), best epoch kept by validation macro F1, then evaluated once on the test set. Loss and validation macro F1 are logged per epoch (training-curve plots in the notebook).

| | A: Image-only | B: Text-only | C: Fused |
|---|---|---|---|
| Encoder(s) | ResNet18 (ImageNet) | DistilBERT (`distilbert-base-uncased`) | both |
| Head | `Linear(512, 37)` | `[CLS]` 768-d → `Linear(768, 37)` | concat 512+768 = 1280 → MLP 1280→256→37 (ReLU, dropout 0.3) |
| Learning rate | 1e-4 | 2e-5 | ResNet 1e-4, DistilBERT 2e-5, MLP 1e-3 |
| Encoder training | fully fine-tuned | fully fine-tuned | both fully fine-tuned |
| Best epoch (val macro F1) | 3 (0.786) | 3 (0.927) | 1 (0.924) |

Per-epoch validation macro F1: A 0.645 → 0.772 → 0.786; B 0.851 → 0.913 → 0.927; C 0.924 → 0.887 → 0.919 (not monotonic).

### Fusion design: early vs late vs joint
- **Early fusion:** the raw (or very low-level) inputs of the modalities are combined before a single model processes them.
- **Late fusion:** each modality is processed by its own encoder, and the resulting representations are combined near the end (here, concatenated) before the classifier.
- **Joint fusion:** the encoders and the fusion/classifier head are trained together end-to-end so the classification loss gradient flows into both encoders.

**What I built:** late fusion by concatenation, as required: ResNet18 and DistilBERT each produce a feature vector, the two are concatenated and a small MLP classifies. Both encoders are fine-tuned together with the head, so training is end-to-end and gradients reach both encoders; the defining design choice is that the modalities are only combined at the feature level, after separate encoding.

### Freezing strategy
All encoders were **fully fine-tuned, nothing frozen**, using small learning rates for the pretrained parts (1e-4 for ResNet, 2e-5 for DistilBERT) so the pretrained weights are not destroyed, and a larger rate (1e-3) for the new MLP head. This is the only configuration I trained; freezing and freeze-then-unfreeze variants were not tried.

---

## 5. Results (test set, 6,657 rows, 37 classes)

| Model | Accuracy | Macro F1 | Weighted F1 | Macro F1 vs text |
|---|---|---|---|---|
| A: Image-only (ResNet18) | 0.9618 | 0.7670 | 0.9597 | −0.1472 |
| B: Text-only (DistilBERT) | 0.9890 | **0.9142** | 0.9882 | 0.0000 |
| C: Fused (late fusion) | 0.9878 | 0.8998 | 0.9870 | −0.0144 |

- Text is the strongest model because titles usually name the item ("Puma Men Grey T-shirt").
- Image-only is the weakest (low-resolution images), and its accuracy (0.96) hides a much lower macro F1 (0.77).
- Fusion is far above image-only but slightly below text-only.

### Training curves
Train/val loss and validation macro F1 per epoch for each model are plotted in the notebook (one figure per model).

### Confusion matrix (fusion)
- Counts view is dominated by the large classes (Topwear 2,309 and Shoes 1,102 test rows), which lie almost perfectly on the diagonal; the row-normalised view shows a clean diagonal for almost all 37 classes.
- The fused model makes **81 errors out of 6,657** test rows (about 1.2%).
- Off-diagonal mass is concentrated in related groups: personal-care classes (Skin, Skin Care, Makeup, Bath and Body, Eyes), Free Gifts (mostly predicted as Watches), and similar pairs (Sandal vs Flip Flops, Dress vs Topwear).

### Per-class F1
Three worst classes for the fused model (lowest F1):

| Class | F1 (A / B / C) | Test rows |
|---|---|---|
| **Bath and Body** | 0.000 / 0.000 / **0.000** | 2 |
| **Skin** | 0.154 / 0.720 / **0.300** | 11 |
| **Free Gifts** | 0.111 / 0.316 / **0.316** | 16 |

(Next is Skin Care with C = 0.462, n = 12.) All of these have very few test rows, so one or two examples move their F1 a lot. Most classes, including all large ones, have F1 above 0.95 for text and fusion (Topwear 0.998, Shoes 0.995, Bags 0.992, Bottomwear 0.989). The image-only model is clearly weaker on small or visually similar classes (e.g. Stoles 0.526, Skin 0.154, Dress 0.784), where text is decisive.

---

## 6. Error analysis

Of the 81 fusion errors on the test set: **41** were also wrong in both single-modality models, **19** were right for text-only (the image misled fusion), and **29** were right for image-only. The notebook shows a grid of eight examples (image, title, true class, fusion / image-only / text-only predictions); the table below describes them, and the explanations are mine, based on what each image and title shows.

| # | Title | True | Fusion | A (img) | B (text) | Likely cause |
|---|---|---|---|---|---|---|
| 1 | Playboy Men Navy Blue Shorts | Loungewear and Nightwear | Bottomwear | Bottomwear | Bottomwear | **Questionable label.** Title and image both show plain shorts; the dataset assigns a nightwear class, a fine distinction no input reveals. |
| 2 | Q&Q Kids Unisex White Dial Watch | Free Gifts | Watches | Watches | Watches | **Questionable label.** Clearly a watch in both image and text; "Free Gifts" is a marketing label, not a product type. |
| 3 | Aspen Women Black Dial Watch | Free Gifts | Watches | Watches | Watches | **Questionable label.** Same as #2; every model agrees with the visible product. |
| 4 | IDEE Men Brown Sunglasses | Free Gifts | Eyewear | Eyewear | Eyewear | **Questionable label.** Obvious sunglasses, labelled Free Gifts. |
| 5 | Streetwear Stay On Mineral Goddess Eye Shadow | Eyes | Makeup | Nails | Eyes | **Image ambiguous.** The small tube looks like nail polish; text alone got it right, the image pulled fusion to Makeup. The classes Eyes/Makeup also overlap. |
| 6 | Revlon Age Defying Sculpt-Logic Night Cream | Skin | Skin Care | Skin Care | Skin | **Overlapping classes + ambiguous image.** Skin and Skin Care are near-synonyms and a cream jar at this resolution does not distinguish them; text alone was right. |
| 7 | Wildcraft Women Blue Printed Wallet | Bags | Wallets | Bags | Wallets | **Misleading text.** The title says "Wallet" but the item is labelled Bags; the image-only model was right and fusion followed the text. |
| 8 | Timberland Women Brown Sandals | Sandal | Shoes | Sandal | Shoes | **Overlapping classes / text uninformative about the class boundary.** The sandal is clear in the image (image-only was right), but text predicted Shoes, which fusion followed. |

Further examples from the error table with the same pattern: Fila "Glide Slippers", Nike "Solarsoft Slide Slippers" and Puma "Duncan Slipper" (true Sandal, predicted Flip Flops: Sandal/Flip Flops overlap), Maxima "Black Watch" (Free Gifts → Watches), Polaroid sunglasses (true Cufflinks → Eyewear, a wrong label), Biotique "Bio Almond Eye Cream" and Olay "Whitening Brightener" (Skin Care / Skin → Makeup).

**Pattern.** Most errors come from questionable or overlapping labels (Free Gifts, Skin vs Skin Care, Sandal vs Flip Flops, Bags vs Wallets), which no model can fix, not from clear model weakness. In 19 cases the image pulled fusion away from a correct text prediction; in the Bags/Wallets and Sandal/Shoes examples the opposite happened (the image was right and fusion followed the text).

---

## 7. Verdict: did fusion help?

**No.** Test macro F1 is 0.8998 for fusion versus 0.9142 for text-only (and 0.7670 for image-only); accuracy is almost identical (0.9878 vs 0.9890).

**By how much, and is it real?** The difference is −0.0144. A bootstrap over the test set (200 resamples) gives a mean difference (fusion − text) of −0.0131 with a 95% interval of [−0.0282, +0.0097]. The interval contains zero, so fusion is **not significantly different** from text-only: it did not help, but we cannot claim it hurt. Caveat: the bootstrap measures test-sampling noise only; each model was trained once, so run-to-run training variance is not measured.

**On which classes?** Fusion and text have (almost) the same F1 on most classes. Fusion is better by at least 0.005 on Eyes (+0.064), Dress (+0.020), Nails (+0.010), Loungewear and Nightwear (+0.007), and worse on **Skin (−0.420)**, **Skin Care (−0.170)**, Makeup (−0.031) and Innerwear (−0.006). Because macro F1 is an average over 37 classes, the Skin class alone costs about 0.011 macro F1 and Skin Care about 0.005; these two small classes (11 and 12 test rows) together account for more than the whole −0.014 gap.

**Why did fusion not help?**
1. Titles already name the category, so text carries almost all of the signal (text-only 0.914 vs image-only 0.767).
2. The images are low resolution, so the extra information they add is weak.
3. In 19 of the 81 fusion errors the text-only model was right, i.e. a visually ambiguous image misled the fused model (e.g. the eye-shadow tube looking like nail polish, creams looking alike).
4. Many remaining errors come from questionable labels (Free Gifts, Skin vs Skin Care, Sandal vs Flip Flops), which no modality fixes.
5. The fused model's best validation epoch was epoch 1 (val macro F1 0.924, then 0.887 and 0.919), a sign that it is slightly less stable than text-only, though with a single run this is only an observation.

**Conclusion.** In this dataset text is the dominant modality, so a zero or slightly negative fusion gain is the expected and honest result rather than a failure of the pipeline.

---

## 8. Limitations

- Each model was trained once (3 epochs); no seeds were varied, so training-run noise is unmeasured.
- Learning rates were chosen per model (1e-4 / 2e-5 / mixed) and not tuned; only full fine-tuning was trained (no frozen or freeze-then-unfreeze variants).
- Several small classes have 2–16 test rows, so their F1 values are unstable.
- The error-analysis explanations are my reading of the images and titles, not verified against the dataset's labelling rules.