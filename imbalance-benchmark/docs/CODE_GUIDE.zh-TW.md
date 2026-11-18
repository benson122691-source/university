# 程式碼閱讀指南（繁體中文）

這份文件告訴你「從哪個檔案開始讀」以及「每個檔案在整個流程中扮演什麼角色」。
每個檔案的最上方都有一段中文說明，函式裡也有逐段註解。

## 建議閱讀順序

| 順序 | 檔案 | 讀完你會知道 |
|---|---|---|
| 1 | `configs/default.yaml` | 實驗有哪些可調的數字（CV 次數、模型、策略、成本假設） |
| 2 | `src/imbalance_lab/data.py` | 資料怎麼載入、信用卡資料有哪些髒資料、怎麼人為製造不同的不平衡比 |
| 3 | `src/imbalance_lab/strategies.py` | 5 種不平衡處理策略各自在做什麼；為什麼要用 SMOTENC 而不是 SMOTE |
| 4 | `src/imbalance_lab/models.py` | 5 個模型、哪些組合不合理要跳過、調參的搜尋範圍 |
| 5 | `src/imbalance_lab/pipeline.py` | **最重要**：4 步驟 Pipeline，以及它如何避免資料洩漏 |
| 6 | `src/imbalance_lab/metrics.py` | 為什麼不平衡問題要看 PR-AUC 而不是只看 ROC-AUC |
| 7 | `src/imbalance_lab/experiments.py` | 調參 → Repeated CV → 最終測試的整體流程 |
| 8 | `src/imbalance_lab/stats.py` | 怎麼判斷「差 0.004」到底是真的還是雜訊 |
| 9 | `src/imbalance_lab/cost.py` | 把機率變成「核准/拒絕」決策、選門檻、敏感度分析 |
| 10 | `src/imbalance_lab/cli.py` | 所有階段如何串起來、結果存到哪裡 |
| 11 | `tests/` | 每個關鍵性質（不洩漏、公式正確）怎麼被自動驗證 |

`plots.py` 只負責畫圖，可以最後再看。

## 整體資料流

```
data.py  ──載入/清理──▶  Dataset
                          │
experiments.split_train_test()   80% train / 20% test（test 鎖起來）
                          │
          ┌───────────────┴───────────────────────────┐
          ▼                                           │
   experiments.tune()      在 train 內部調參            │
          ▼                                           │
   experiments.run_cv()    5×3 Repeated CV              │
     ├─ 每一折：pipeline.build_pipeline() → fit → predict
     ├─ 分數 → stats.py 做檢定、排名                     │
     └─ out-of-fold 機率 → 留給 cost.py 選門檻           │
          ▼                                           ▼
   experiments.evaluate_test()   用全部 train 訓練，在 test 上評估一次
          ▼
   cost.py   三種門檻規則 × 成本敏感度分析 × 機率校準
```

## 幾個關鍵觀念（面試很可能被問）

**1. 什麼是資料洩漏 (data leakage)？這個專案怎麼避免？**
驗證/測試資料的資訊「偷偷」進入訓練過程，讓分數虛高。
常見錯誤：先對全部資料做 SMOTE 或標準化，再切 train/test。
本專案把「標準化 + 重採樣 + 編碼 + 模型」全部包進 imbalanced-learn 的 `Pipeline`，
交叉驗證時每一折都重新 fit，而且 `Pipeline` 在 `predict` 時會自動跳過 sampler。
`tests/test_pipeline.py::test_sampler_only_sees_training_fold` 用一個「間諜 sampler」證明這件事。

**2. 為什麼不能用一般的 paired t-test 比較 CV 分數？**
不同折的訓練集大量重疊，分數彼此相關，一般 t-test 會低估變異 → 容易誤判顯著。
本專案用 Nadeau–Bengio 修正版（`stats.corrected_resampled_ttest`），再用 Holm 校正多重比較。

**3. 欠採樣和「調門檻」有什麼關係？**
欠採樣讓模型看到的違約比例變高，預測機率整體偏高，效果類似把門檻調低。
但調門檻不用丟掉任何資料。`cost.correct_undersampled_proba` 可以把欠採樣模型的機率換算回真實比例。

**4. 為什麼主要指標是 PR-AUC？**
ROC 曲線的 FPR 分母是龐大的負類，誤報很多 FPR 也不大，曲線容易「看起來很好」。
PR 曲線只看「被判成正類的人裡有多少真的是正類」，不會被負類數量稀釋。

## 怎麼自己跑

```bash
pip install -e ".[dev]"
pytest -q                                         # 先確認測試全過（幾秒鐘）
imbalance-lab run --config configs/quick.yaml     # 快速版，約 2 分鐘，確認流程能跑完
imbalance-lab run                                 # 完整實驗（2 核心約 40 分鐘）
imbalance-lab run --stages cost                   # 改了 configs 裡的成本假設，只重跑成本分析
```
