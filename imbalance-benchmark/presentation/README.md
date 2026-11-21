# 簡報

`imbalance-benchmark-deck.pptx`：15 分鐘面試簡報（繁體中文，16 頁），每頁的講稿在 PowerPoint 的「備忘稿」。

簡報上的數字全部從 `results/` 自動讀取，不手打。結果更新後重新產生：

```bash
cd imbalance-benchmark
python presentation/prepare_data.py   # 整理數字 + 產生簡報專用的大字版圖片（img/）
PPTX_SKILL_DIR=<pptx skill 資料夾> node presentation/build_deck.js
```

`build_deck.js` 需要 Node.js 套件 `pptxgenjs`、`react`、`react-dom`、`react-icons`、`sharp`，
以及 pptx skill 提供的 `apply_theme.js`（寫入主題色）。字型使用「微軟正黑體」。
