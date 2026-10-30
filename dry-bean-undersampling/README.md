# Undersampling for Imbalanced Classification

A multi-dataset pipeline that evaluates undersampling techniques for imbalanced
classification, comparing **KNN, Random Forest, SVM, XGBoost** classifiers across
**SMOTE / Undersampling / Hyperparameter Optimization** methods.

The pipeline is shared across domains to validate the generality of the
undersampling strategy:

| Dataset | Domain | Samples | Features | Classes | Imbalance ratio |
|---|---|---|---|---|---|
| [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | Agriculture: bean images | 13,611 | 16 | 7 (multiclass) | ≈ 6.8 : 1 |
| [Taiwan Credit Card Default](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | Finance: credit card default | 30,000 | 23 | 2 (binary) | ≈ 3.5 : 1 |

## Dataset 1: Dry Bean Dataset (UCI)

- 13,611 samples, 16 features, 7 classes
- Class distribution (imbalanced):

| Class | Count | Proportion |
|---|---|---|
| DERMASON | 3,546 | 26.05% |
| SIRA | 2,636 | 19.37% |
| SEKER | 2,027 | 14.89% |
| HOROZ | 1,928 | 14.17% |
| CALI | 1,630 | 11.98% |
| BARBUNYA | 1,322 | 9.71% |
| BOMBAY | 522 | 3.84% |

Source: Koklu, M., & Ozkan, I. A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques.* Computers and Electronics in Agriculture, 174, 105507.

## Dataset 2: Taiwan Credit Card Default (UCI, finance)

- 30,000 samples, 23 features, binary target (`default payment next month`)
- Class distribution (imbalanced):

| Class | Count | Proportion |
|---|---|---|
| Non-default (0) | 23,364 | 77.88% |
| Default (1) | 6,636 | 22.12% |

Source: Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480.

## Methodology

1. **Data selection**: choose a dataset (`--dataset dry_bean` / `--dataset credit_card`);
   both run the exact same pipeline to validate cross-domain generality
2. **Split**: 80% training / 20% test (stratified, `random_state=42`)
3. **Standardization**: `StandardScaler` (fit on the training set only; the same
   parameters are applied to the test set)
4. **Undersampling**:
   - **Scheme A (majority-only)**: reduce the majority class to a given ratio of
     the minority count (1:1 ~ 1:5)
   - **Scheme B (all non-minority)**: reduce every non-minority class to the
     minority count
   - Note: for binary data (credit_card) schemes A and B coincide mathematically
5. **Classifiers × methods**:

| Method | Description |
|---|---|
| SMOTE | Synthesizes minority-class samples (oversampling) |
| Undersampling | Random undersampling (Scheme A, ratio 1:1) + default parameters |
| Optimized | Tuned hyperparameters, no resampling |
| Undersampling+Optimized | Random undersampling (Scheme A) + tuned parameters |

6. **Metrics**: Precision, Recall, F1-Score, ROC AUC (macro average)

### Tuned classifier hyperparameters

| Classifier | Best parameters |
|---|---|
| SVM | `C=10, gamma=0.1, kernel='rbf'` |
| Random Forest | `max_depth=None, max_features='log2', min_samples_leaf=1, min_samples_split=10, n_estimators=100` |
| KNN | `metric='euclidean', n_neighbors=9, weights='distance'` |
| XGBoost | `colsample_bytree=1.0, learning_rate=0.2, max_depth=6, n_estimators=100, subsample=1.0` |

## Installation & Usage

```bash
# Recommended: use a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python undersampling_analysis.py                      # all datasets
python undersampling_analysis.py --dataset dry_bean   # one dataset only
```

On first run the data is downloaded automatically from UCI into `data/`
(dry_bean ≈ 1 MB, credit_card ≈ 2 MB). If a matching local file already exists
(`Dry_Bean_Dataset.xlsx`, `default of credit card clients.xls`, or a CSV), it is
used instead.

## Output

```
figures/<dataset>/
├── 01_class_distribution.png        # class-distribution bar chart (with proportions)
├── 02_roc_optimized.png             # ROC curves (optimized parameters)
├── 03_roc_undersample_optimized.png # ROC curves (undersampling + optimized)
├── 04_f1_method_comparison.png      # F1 comparison (methods × classifiers)
├── 05_scheme_ab_comparison.png      # Scheme A vs Scheme B
└── 06_ratio_analysis.png            # undersampling ratio analysis (1:1 ~ 1:5)

results/<dataset>/
├── class_distribution.csv
├── method_comparison_f1.csv         # 4 classifiers × 4 methods (P/R/F1/AUC)
├── scheme_ab_comparison.csv
└── ratio_analysis.csv
```

## Key Findings

### Dry Bean Dataset (agriculture, 7 classes, imbalance ≈ 6.8:1)

- **Full undersampling (Scheme B)** significantly improves KNN and SVM
  (measured SVM Scheme B F1 ≈ 0.932).
- **Partial undersampling (Scheme A)** helps XGBoost and Random Forest while
  retaining more information and lower training cost.
- **Best undersampling ratio** ≈ **1:4 ~ 1:5** (F1 ≈ 0.935); the ratio range
  1:3 ~ 1:5 performs best overall.

### Taiwan Credit Card Default (finance, binary, imbalance ≈ 3.5:1)

- **SMOTE and hyperparameter optimization beat undersampling** on the financial
  data: SVM×SMOTE F1 ≈ 0.698 and Random Forest×Optimized F1 ≈ 0.696 are the best
  combinations.
- **Undersampling helps little with mild imbalance**: aggressive undersampling
  (1:1) drops XGBoost F1 to ≈ 0.66; the best ratio is **1:2 (F1 ≈ 0.70)** —
  removing too much majority data loses information.
- **Scheme A ≈ Scheme B**: for binary data the two schemes coincide, as expected.
- **Cross-domain conclusion**: undersampling suits **severely imbalanced
  multiclass** problems (Dry Bean 6.8:1); for **mildly imbalanced binary**
  problems (credit card 3.5:1), SMOTE / hyperparameter optimization is preferable
  — the choice of technique should depend on the imbalance severity.

> Note: the pipeline uses `random_state=42` for reproducibility; individual
> numbers may differ slightly from any external report (sampling randomness),
> while the overall conclusions remain stable.

## File Structure

```
dry-bean-undersampling/
├── undersampling_analysis.py   # main pipeline
├── requirements.txt
├── README.md
└── .gitignore
```
