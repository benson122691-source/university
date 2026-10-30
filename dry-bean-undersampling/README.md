# 欠採樣於不平衡分類分析 (Undersampling for Imbalanced Classification)

畢業專題（期末報告）程式碼 — 探討欠採樣技術在多類別不平衡分類任務中的效果，並比較
**KNN、隨機森林、SVM、XGBoost** 四種分類器與 **SMOTE / 欠採樣 / 超參數最佳化** 等處理方法。

支援多資料集，驗證策略在不同領域的通用性（簡報「未來展望」：更廣泛的資料集驗證）：

| 資料集 | 領域 | 筆數 | 特徵 | 類別 | 不平衡比例 |
|---|---|---|---|---|---|
| [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | 農業：豆類影像 | 13,611 | 16 | 7 類（多分類） | ≈ 6.8 : 1 |
| [Taiwan Credit Card Default](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | 金融：信用卡違約 | 30,000 | 23 | 2 類（二元） | ≈ 3.5 : 1 |

## 資料集 1：Dry Bean Dataset（UCI）

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

## 資料集 2：Taiwan Credit Card Default（UCI，金融）

- 30,000 筆、23 個特徵、二元分類（`default payment next month`）
- 類別分布（不平衡）：

| 類別 | 筆數 | 比例 |
|---|---|---|
| 非違約（0） | 23,364 | 77.88% |
| 違約（1） | 6,636 | 22.12% |

資料來源：Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480.

## 研究方法流程

1. **資料選取**：可指定資料集（`--dataset dry_bean` / `--dataset credit_card`），
   兩資料集皆執行完全相同的研究流程，以驗證策略的跨領域通用性
2. **資料分割**：80% 訓練集 / 20% 測試集（分層抽樣，`random_state=42`）
3. **標準化**：`StandardScaler`（僅以訓練集擬合，測試集沿用相同參數）
4. **欠採樣**：
   - **方案 A（僅對最多數組）**：將最多數類別降至最少數類別數量的指定比例（1:1 ~ 1:5）
   - **方案 B（對所有非最少組）**：所有類別全部降至最少數組數量
   - 註：二元分類（credit_card）時方案 A 與方案 B 數學上相同（兩類即為最少/最多組）
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
python undersampling_analysis.py                      # 執行全部資料集
python undersampling_analysis.py --dataset dry_bean   # 僅執行指定資料集
```

首次執行會自動從 UCI 下載資料集到 `data/`（dry_bean ≈ 1 MB、credit_card ≈ 2 MB）；
若 `data/` 下已有對應檔案（`Dry_Bean_Dataset.xlsx`、`default of credit card clients.xls`
或 CSV），則直接使用本機檔案。

## 輸出

```
figures/<資料集>/
├── 01_class_distribution.png        # 類別分布長條圖（含比例）
├── 02_roc_optimized.png             # ROC 曲線比較（最佳化參數）
├── 03_roc_undersample_optimized.png # ROC 曲線比較（欠採樣＋最佳化）
├── 04_f1_method_comparison.png      # F1 比較（方法 × 分類器）
├── 05_scheme_ab_comparison.png      # 方案 A vs 方案 B
└── 06_ratio_analysis.png            # 欠採樣比例分析（1:1 ~ 1:5）

results/<資料集>/
├── class_distribution.csv
├── method_comparison_f1.csv         # 四分類器 × 四方法（P/R/F1/AUC）
├── scheme_ab_comparison.csv
└── ratio_analysis.csv
```

## 主要結論

### Dry Bean Dataset（農業，7 類別，不平衡 ≈ 6.8:1）— 簡報主體

- **完全欠採樣（方案 B）**：顯著提升 KNN、SVM 的性能（實測 SVM 方案 B F1 ≈ 0.932）。
- **部分欠採樣（方案 A）**：僅對最多數組欠採樣，對 XGBoost 與隨機森林有提升，
  且保留較多資訊、訓練成本較低。
- **最佳欠採樣比例**：簡報結論約 **1:3 ~ 1:4**；本實測 1:4、1:5 最佳（F1 ≈ 0.935），
  兩者皆落在 1:3 ~ 1:5 區間。

### Taiwan Credit Card Default（金融，二元，不平衡 ≈ 3.5:1）— 未來展望：跨領域驗證

- 金融資料集上 **SMOTE 與超參數最佳化優於欠採樣**：SVM×SMOTE F1 ≈ 0.698、
  Random Forest×最優化 F1 ≈ 0.696 為最佳組合。
- **欠採樣對輕度不平衡幫助有限**：過度欠採樣（1:1）使 XGBoost F1 降至 ≈ 0.66；
  最佳比例為 **1:2（F1 ≈ 0.70）**，過度刪除多數類反而損失資訊。
- **方案 A ≈ 方案 B**：二元分類時兩方案數學上相同，實測結果符合預期。
- **跨領域結論**：欠採樣較適合**嚴重不平衡的多分類**（Dry Bean 6.8:1）；
  對**輕度不平衡的二元**問題（信用卡 3.5:1）以 SMOTE／超參數最佳化為佳——
  處理技術的選擇需依資料集不平衡程度而定，驗證了簡報「更廣泛資料集驗證」的價值。

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
