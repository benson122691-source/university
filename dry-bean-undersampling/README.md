# 欠採樣於不平衡分類分析 (Undersampling for Imbalanced Classification)

畢業專題（期末報告）程式碼 — 探討欠採樣技術在多類別不平衡分類任務中的效果，並比較
**KNN、隨機森林、SVM、XGBoost** 四種分類器與 **SMOTE / 欠採樣 / 超參數最佳化** 等處理方法。

## 資料集：Dry Bean Dataset（UCI）

- 13,611 筆、16 個特徵、7 個類別
- 類別分布（不平衡）：

| 類別 | 筆數 | 比例 |
|---|---|---|
| DERMASON | 3,546 | 26.05% |
| SIRA | 2,636 | 19.37% |
| SEKER | 2,027 | 14.89% |
| HOROZ | 1,928 | 14.17% |
| CALI | 1,630 | 11.98% |
| BARBUNYA | 1,322 | 9.71% |
| BOMBAY | 522 | 3.84% |

資料來源：Koklu, M., & Ozkan, I. A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques.* Computers and Electronics in Agriculture, 174, 105507.  
UCI：https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

## 研究方法流程

1. **資料選取**：Dry Bean Dataset
2. **資料分割**：80% 訓練集 / 20% 測試集（分層抽樣，`random_state=42`）
3. **標準化**：`StandardScaler`（僅以訓練集擬合，測試集沿用相同參數）
4. **欠採樣**：
   - **方案 A（僅對最多數組）**：將最多數類別降至最少數類別數量的指定比例（1:1 ~ 1:5）
   - **方案 B（對所有非最少組）**：所有類別全部降至最少數組數量
5. **分類器與方法組合**：

| 方法 | 說明 |
|---|---|
| SMOTE | 合成少數類樣本（過採樣） |
| 下採樣 | 隨機欠採樣（方案 A，比例 1:1）+ 預設參數 |
| 最優化 | 超參數最佳化（簡報參數），不重新取樣 |
| 下採樣+最優化 | 隨機欠採樣（方案 A）＋ 最佳化參數 |

6. **評估指標**：Precision、Recall、F1-Score、ROC AUC（macro 平均）

### 分類器最佳化參數（簡報）

| 分類器 | 最佳參數 |
|---|---|
| SVM | `C=10, gamma=0.1, kernel='rbf'` |
| Random Forest | `max_depth=None, max_features='log2', min_samples_leaf=1, min_samples_split=10, n_estimators=100` |
| KNN | `metric='euclidean', n_neighbors=9, weights='distance'` |
| XGBoost | `colsample_bytree=1.0, learning_rate=0.2, max_depth=6, n_estimators=100, subsample=1.0` |

## 安裝與執行

```bash
# 建議使用虛擬環境
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python undersampling_analysis.py
```

首次執行會自動從 UCI 下載資料集到 `data/`（約 1 MB）；若 `data/` 下已有
`Dry_Bean_Dataset.xlsx` 或 `DryBean_Dataset.csv`（Kaggle 格式），則直接使用本機檔案。

## 輸出

```
figures/
├── 01_class_distribution.png        # 類別分布長條圖（含比例）
├── 02_roc_optimized.png             # ROC 曲線比較（最佳化參數）
├── 03_roc_undersample_optimized.png # ROC 曲線比較（欠採樣＋最佳化）
├── 04_f1_method_comparison.png      # F1 比較（方法 × 分類器）
├── 05_scheme_ab_comparison.png      # 方案 A vs 方案 B
└── 06_ratio_analysis.png            # 欠採樣比例分析（1:1 ~ 1:5）

results/
├── class_distribution.csv
├── method_comparison_f1.csv         # 四分類器 × 四方法（P/R/F1/AUC）
├── scheme_ab_comparison.csv
└── ratio_analysis.csv
```

## 主要結論（詳見簡報）

- **完全欠採樣（方案 B）**：顯著提升 KNN、SVM 的性能（SVM 最佳 F1 ≈ 0.93）。
- **部分欠採樣（方案 A）**：僅對最多數組欠採樣，對 XGBoost 與隨機森林有提升，
  且保留較多資訊、訓練成本較低。
- **模型選擇**：嚴重不平衡建議 SVM＋完全欠採樣；較低不平衡可選對不平衡敏感度較低的
  XGBoost / 隨機森林。
- **最佳欠採樣比例**：約 **1:3 ~ 1:4**，在保留性能與數據特徵之間取得最佳平衡。

> 注意：本程式以 `random_state=42` 重現流程，個別數值可能與簡報截圖有微小差異
> （取樣隨機性所致），整體結論一致。

## 檔案結構

```
dry-bean-undersampling/
├── undersampling_analysis.py   # 主程式（完整研究流程）
├── requirements.txt
├── README.md
└── .gitignore
```
