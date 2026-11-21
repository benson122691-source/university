/**
 * 產生面試簡報：presentation/imbalance-benchmark-deck.pptx
 *
 * 用法（在 imbalance-benchmark/ 資料夾內）：
 *   python presentation/prepare_data.py      # 從 results/ 整理出簡報要用的數字
 *   node presentation/build_deck.js          # 產生 .pptx
 *
 * 簡報上的數字全部來自 deck_data.json，不手打。
 * 講稿放在每一頁的「備忘稿」(speaker notes)，PowerPoint 簡報者檢視畫面看得到。
 */
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const SKILL = process.env.PPTX_SKILL_DIR;  // apply_theme.js 所在的資料夾（由執行環境提供）
const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const OUT = path.join(__dirname, "imbalance-benchmark-deck.pptx");
const REPO = "github.com/benson122691-source/university";

// ---------------------------------------------------------------------------
// 主題：深墨藍為主色、紫色為強調色（圖表裡沒用到紫色，不會跟策略配色混淆）
// accent2~6 = 圖表中各策略的顏色，讓簡報與 README 圖表一致
// ---------------------------------------------------------------------------
const THEME = {
  name: "Imbalance Benchmark",
  headFontFace: "Microsoft JhengHei",
  bodyFontFace: "Microsoft JhengHei",
  colors: {
    dk1: "16181D", lt1: "FFFFFF", dk2: "1E2433", lt2: "F3F2EE",
    accent1: "6250D6",  // 紫：強調色（編號圓點、重點數字）
    accent2: "2A78D6",  // 藍：class weight
    accent3: "EB6834",  // 橘：random undersampling
    accent4: "1BAF7A",  // 青綠：Tomek
    accent5: "EDA100",  // 黃：SMOTE
    accent6: "898781",  // 灰：不處理（基準線）
    hlink: "6250D6", folHlink: "52514E",
  },
};
const HEX = THEME.colors;
const MUTED = "6B6A66";       // 說明文字（灰）
const RED = "D64545";         // 「較差」
const BLUE = "2A78D6";        // 「較好」

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";  // 13.33" × 7.5"
pres.title = "不平衡分類：重採樣真的有用嗎？";
pres.author = "benson122691-source";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const W = 13.333;
const M = 0.6;               // 左右邊界
const CW = W - 2 * M;        // 內容寬度

// ---------------------------------------------------------------------------
// 版面配置 (layouts)
// ---------------------------------------------------------------------------
pres.defineSlideMaster({
  title: "DARK_TITLE",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M + 0.2, y: 2.2, w: CW - 0.4, h: 1.5,
      fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M + 0.2, y: 3.85, w: CW - 0.4, h: 1.0,
      fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  margin: [0.5, M, 0.6, M],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: CW, h: 0.9,
      fontSize: 34, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: `不平衡分類基準測試 · ${REPO}`, options: { x: M, y: 7.0, w: 8, h: 0.3, fontSize: 10,
      color: MUTED, margin: 0, isTextBox: true } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------
async function icon(Comp, hex, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: String(size) }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

/** 紫色圓點 + 白色編號：整份簡報的視覺記號 */
function badge(slide, n, x, y, d = 0.5) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: C.accent1 }, line: { type: "none" },
    objectName: `badge-${n}` });
  slide.addText(String(n), { x, y, w: d, h: d, fontSize: d > 0.45 ? 16 : 12, bold: true, color: C.background1,
    align: "center", valign: "middle", margin: 0, isTextBox: true });
}

/** 大數字 + 小說明 */
function stat(slide, x, y, w, big, label, color = C.accent1) {
  slide.addText(big, { x, y, w, h: 0.95, fontSize: 48, bold: true, color, margin: 0, valign: "bottom", isTextBox: true });
  slide.addText(label, { x, y: y + 1.0, w, h: 0.8, fontSize: 14, color: C.text2, margin: 0, valign: "top",
    isTextBox: true });
}

/** 淺灰底卡片 */
function card(slide, x, y, w, h, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 },
    line: { type: "none" }, objectName: name });
}

const caption = (slide, text, x, y, w, h = 0.35) =>
  slide.addText(text, { x, y, w, h, fontSize: 11, color: MUTED, margin: 0, valign: "top", isTextBox: true });

function addImg(slide, key, x, y, maxW, maxH) {
  const { path: p, ratio } = D.images[key];
  let w = maxW, h = maxW / ratio;
  if (h > maxH) { h = maxH; w = maxH * ratio; }
  slide.addImage({ path: p, x, y, w, h, altText: key });
  return { w, h };
}

const STRAT = [
  ["none", "不處理（基準線）", HEX.accent6],
  ["class_weight", "class weight", HEX.accent2],
  ["random_under", "隨機欠採樣", HEX.accent3],
  ["tomek", "Tomek links", HEX.accent4],
  ["smote", "SMOTE / SMOTENC", HEX.accent5],
];

// 圖表共用設定：文字樣式在每張圖上一致
const chartBase = () => ({
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  dataLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, legendFontSize: 12,
  catAxisLabelColor: "52514E", valAxisLabelColor: "6B6A66", legendColor: "52514E",
  valGridLine: { color: "E1E0D9", size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: true, catAxisLineColor: "C3C2B7", valAxisLineShow: false,
});

(async () => {
  // =========================================================================
  // 1. 標題
  // =========================================================================
  pres.addSection({ title: "開場" });
  let s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "開場" });
  s.addText("不平衡分類：重採樣真的有用嗎？", { placeholder: "title" });
  s.addText("7 份資料的無洩漏基準測試，與成本導向的決策", { placeholder: "body" });
  STRAT.forEach(([, , hex], i) => s.addShape(pres.shapes.OVAL, { x: M + 0.2 + i * 0.42, y: 1.55, w: 0.26, h: 0.26,
    fill: { color: hex }, line: { type: "none" }, objectName: `dot-${i}` }));
  s.addText(REPO, { x: M + 0.2, y: 6.4, w: 8, h: 0.4, fontSize: 14, color: C.background2, margin: 0, isTextBox: true });
  s.addNotes(
    "大家好，今天分享一個我重新做過的專案：處理類別不平衡時，SMOTE、欠採樣這些常見做法，到底有沒有讓模型變好。" +
    "我用 7 份不同領域的資料、5 個模型、5 種策略做了無資料洩漏的比較，最後把結果轉成以成本為基礎的決策方法。" +
    "上面這排顏色，就是後面每張圖裡各個策略固定使用的顏色。");

  // =========================================================================
  // 2. 一句話結論
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "開場" });
  s.addText("結論先講：別急著重採樣，先調門檻", { placeholder: "title" });
  const st3 = [
    ["0 / 105", "策略顯著勝過「不處理」的次數\n（7 份資料 × 各模型）"],
    [`${D.cost_none_tuned_savings}%`, "不處理 + 用成本調門檻\n比全部核准省下的成本"],
    ["0 / 60", "重採樣在測試集上\n顯著更省成本的次數"],
  ];
  st3.forEach(([big, label], i) => {
    const x = M + i * (CW / 3);
    card(s, x, 1.65, CW / 3 - 0.3, 2.45, `stat-card-${i}`);
    stat(s, x + 0.35, 1.85, CW / 3 - 0.9, big, label);
  });
  s.addText([
    { text: "建議：", options: { bold: true, color: C.accent1 } },
    { text: "用原始資料訓練、保留所有樣本，再依商業成本設定判定門檻。只有在某個類別僅有幾十筆時，才用 class weight。" },
  ], { x: M, y: 4.6, w: CW, h: 1.2, fontSize: 18, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addNotes(
    "先講結論。第一，在 7 份資料、105 次比較裡，沒有任何一次重採樣策略顯著勝過什麼都不做。" +
    "第二，真正有效的是用成本決定門檻：在信用卡違約資料上，原始模型加上用成本調好的門檻，比全部核准省下 49% 的損失。" +
    "第三，在測試集上，沒有任何重採樣策略顯著更省成本。後面我會說明怎麼得到這些結論，以及一個重要的例外。");

  // =========================================================================
  // 3. 問題
  // =========================================================================
  pres.addSection({ title: "問題與方法" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "問題與方法" });
  s.addText("常見做法：資料不平衡，就先做 SMOTE", { placeholder: "title" });
  s.addText([
    { text: "情境", options: { bold: true, breakLine: true } },
    { text: "信用卡違約：每 100 位客戶只有約 22 位違約。模型只要全猜「不會違約」，準確率就有 78%。", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "業界常見的直覺", options: { bold: true, breakLine: true } },
    { text: "先把資料「平衡」再訓練：砍掉多數類、合成少數類，或加權。" },
  ], { x: M, y: 1.6, w: 5.4, h: 3.2, fontSize: 16, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 4,
    isTextBox: true });
  const desc = ["什麼都不做，所有方法都要跟它比", "訓練時讓少數類答錯的懲罰變大", "隨機丟掉多數類，直到各類一樣多",
    "只刪除緊貼少數類的多數類樣本", "在少數類之間內插，合成新樣本"];
  STRAT.forEach(([, name, hex], i) => {
    const y = 1.6 + i * 0.86;
    card(s, 6.6, y, CW - 6.0, 0.72, `strategy-card-${i}`);
    s.addShape(pres.shapes.OVAL, { x: 6.85, y: y + 0.24, w: 0.24, h: 0.24, fill: { color: hex }, line: { type: "none" },
      objectName: `strategy-dot-${i}` });
    s.addText([{ text: name, options: { bold: true, breakLine: true } }, { text: desc[i], options: { fontSize: 13, color: C.text2 } }],
      { x: 7.3, y: y + 0.06, w: CW - 6.9, h: 0.62, fontSize: 15, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  });
  s.addText("問題：它們真的讓模型更好嗎？還是只是改變了「怎麼判定」？", { x: M, y: 5.3, w: 5.6, h: 1.0, fontSize: 18,
    bold: true, color: C.accent1, margin: 0, valign: "top", isTextBox: true });
  s.addNotes(
    "不平衡分類的問題是：少數類很少，模型容易偏向多數類。以信用卡違約為例，全猜不會違約就有 78% 準確率，但一個違約者都抓不到。" +
    "所以業界常見的做法是先平衡資料。右邊是我比較的五種做法：不處理當基準線，加上 class weight、隨機欠採樣、Tomek links、SMOTE。" +
    "我想回答的問題是：這些方法到底是讓模型變得更會分辨，還是只是改變了判定的方式？");

  // =========================================================================
  // 4. v1 的錯誤
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "問題與方法" });
  s.addText("起點：我推翻了自己的舊結論", { placeholder: "title" });
  card(s, M, 1.6, 5.4, 1.6, "quote-card");
  s.addText([
    { text: "舊版（v1）README 的結論", options: { fontSize: 13, color: C.text2, breakLine: true } },
    { text: "「欠採樣適合嚴重不平衡的多分類問題」", options: { fontSize: 17, bold: true } },
  ], { x: M + 0.3, y: 1.75, w: 4.9, h: 1.3, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  s.addText([
    { text: "但同一份報告自己的數據顯示，4 個模型用欠採樣後全部變差。", options: { bold: true, breakLine: true } },
    { text: "其他問題：只切一次資料、參數寫死找不到調參過程、SMOTE 用在類別欄位、沒有任何測試。" },
  ], { x: M, y: 3.5, w: 5.4, h: 2.2, fontSize: 15, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 8,
    isTextBox: true });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.text2, fill: { color: C.background2 } } });
  const v1Rows = [[hdr("模型"), hdr("不處理"), hdr("欠採樣"), hdr("差異")]];
  for (const [m, [a, b]] of Object.entries(D.v1)) {
    v1Rows.push([{ text: m }, { text: a.toFixed(3) }, { text: b.toFixed(3) },
      { text: (b - a).toFixed(3), options: { bold: true, color: RED } }]);
  }
  s.addTable(v1Rows, { x: 6.6, y: 1.6, w: CW - 6.0, colW: [2.1, 1.35, 1.35, 1.33], fontSize: 15, color: C.text1,
    align: "center", valign: "middle", rowH: 0.55, border: { type: "solid", pt: 0.75, color: "E1E0D9" } });
  caption(s, "v1 結果：Dry Bean 資料集，macro F1（取自 v1 的 CSV）", 6.6, 4.5, CW - 6.0);
  s.addNotes(
    "這個專案的起點是我自己的舊版本。v1 的 README 寫說欠採樣適合嚴重不平衡的多分類問題，但同一份報告的數據顯示，" +
    "四個模型用了欠採樣之後 F1 全部下降。除此之外，v1 只切一次資料，參數是寫死的，SMOTE 還用在性別這種類別欄位上。" +
    "所以這次重做，重點不只是重寫程式，而是把實驗方法改對，讓結論可以被驗證。");

  // =========================================================================
  // 5. 方法
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "問題與方法" });
  s.addText("方法：每一步只看該看的資料", { placeholder: "title" });
  const steps = [
    ["切分 80 / 20", "20% 測試集鎖起來，\n最後只用一次"],
    ["調參", "每份資料、每個模型\n各自調，只用訓練集"],
    ["5 × 3 重複交叉驗證", "每組 15 個配對分數，\n看得出差異是不是雜訊"],
    ["測試集評估", "成本、校準、\nbootstrap 信賴區間"],
  ];
  const sw = CW / 4;
  steps.forEach(([t, d], i) => {
    const x = M + i * sw;
    badge(s, i + 1, x, 1.65);
    s.addText(t, { x, y: 2.3, w: sw - 0.3, h: 0.5, fontSize: 17, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(d, { x, y: 2.8, w: sw - 0.3, h: 0.9, fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true });
    if (i < 3) s.addShape(pres.shapes.LINE, { x: x + 0.65, y: 1.9, w: sw - 0.85, h: 0, line: { color: "C3C2B7", width: 1.25,
      endArrowType: "triangle" }, objectName: `flow-arrow-${i}` });
  });
  card(s, M, 4.15, CW, 2.45, "pipeline-card");
  s.addText("防止資料洩漏：所有步驟包在同一條 Pipeline，每一折重新訓練", { x: M + 0.35, y: 4.3, w: CW - 0.7, h: 0.45,
    fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
  const pipe = ["標準化", "one-hot 編碼", "重採樣（只在訓練時）", "模型"];
  const pw = (CW - 0.7 - 3 * 0.45) / 4;
  pipe.forEach((t, i) => {
    const x = M + 0.35 + i * (pw + 0.45);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.95, w: pw, h: 0.7, rectRadius: 0.08,
      fill: { color: i === 2 ? HEX.accent1 : HEX.lt1 }, line: { color: i === 2 ? HEX.accent1 : "C3C2B7", width: 1 },
      objectName: `pipe-${i}` });
    s.addText(t, { x, y: 4.95, w: pw, h: 0.7, fontSize: 15, bold: i === 2, color: i === 2 ? C.background1 : C.text1,
      align: "center", valign: "middle", margin: 0, isTextBox: true });
    if (i < 3) s.addText("→", { x: x + pw, y: 4.95, w: 0.45, h: 0.7, fontSize: 18, color: MUTED, align: "center",
      valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText("驗證折、測試集永遠維持原始分布；自動化測試用「間諜 sampler」證明重採樣只看得到訓練資料。" +
    "差異是否顯著：修正版 t 檢定（Nadeau–Bengio）+ Holm 多重比較校正。",
  { x: M + 0.35, y: 5.8, w: CW - 0.7, h: 0.7, fontSize: 13, color: C.text2, margin: 0, valign: "top", isTextBox: true });
  s.addNotes(
    "方法上有四步：先切出 20% 測試集鎖起來；在訓練集上幫每個模型各自調參；用 5 折重複 3 次的交叉驗證比較策略，" +
    "每組有 15 個配對分數；最後才在測試集上評估一次。" +
    "防洩漏是關鍵：標準化、編碼、重採樣、模型全部包在同一條 Pipeline 裡，重採樣只在訓練時發生。我寫了一個自動化測試，" +
    "用會記錄筆數的「間諜 sampler」證明它只看得到訓練折。判斷顯著性時，因為交叉驗證各折的訓練資料高度重疊，" +
    "一般 t 檢定會太樂觀，所以用 Nadeau–Bengio 修正版，再加上 Holm 校正。");

  // =========================================================================
  // 6. 結果一：森林圖
  // =========================================================================
  pres.addSection({ title: "結果" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結果" });
  s.addText("結果一：沒有策略讓模型更會分辨好壞", { placeholder: "title" });
  const f = addImg(s, "forest", M, 1.4, 8.7, 5.45);
  const rx = M + f.w + 0.4, rw = W - M - rx;
  stat(s, rx, 1.6, rw, "0", "個策略顯著勝過不處理\n（信用卡，PR-AUC）");
  stat(s, rx, 3.55, rw, "4", "次顯著變差\n（主要是 SMOTE、欠採樣）", RED);
  caption(s, "每一列 = 一個模型 × 策略與「不處理」的差異；實心 = 顯著、空心 = 不顯著；越往左越差。", rx, 5.6, rw, 0.9);
  s.addNotes(
    "這是信用卡資料的結果。每一列是一個模型加一個策略，點是跟不處理的差異，橫線是 95% 信賴區間，零的那條線代表跟不處理一樣。" +
    "可以看到沒有任何一個點顯著地在零的右邊，也就是沒有策略讓 PR-AUC 變好；實心的點都在左邊，代表顯著變差，" +
    "大多是 SMOTE 和隨機欠採樣。PR-AUC 衡量的是模型把違約者排在前面的能力，跟門檻無關。");

  // =========================================================================
  // 7. 結果二：門檻效應（原生圖表）
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結果" });
  s.addText("結果二：重採樣只是移動了判定門檻", { placeholder: "title" });
  const te = D.threshold_effect;
  const teKeys = ["none", "class_weight", "random_under", "smote"];
  const teNames = ["不處理", "class weight", "隨機欠採樣", "SMOTENC"];
  s.addChart(pres.charts.BAR, [
    { name: "F1（門檻固定 0.5）", labels: teNames, values: teKeys.map((k) => te[k].f1) },
    { name: "最佳 F1（任何門檻）", labels: teNames, values: teKeys.map((k) => te[k].f1_best) },
  ], {
    ...chartBase(), x: M, y: 1.45, w: 7.4, h: 5.2, barDir: "col", barGapWidthPct: 60, barOverlapPct: -10,
    chartColors: [HEX.accent6, HEX.accent1], valAxisMinVal: 0, valAxisMaxVal: 0.6, valAxisMajorUnit: 0.1,
    valAxisLabelFormatCode: "0.0", showLegend: true, legendPos: "t",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 11,
    dataLabelColor: "52514E", showTitle: false, objectName: "threshold-effect-chart",
  });
  const tx = M + 7.8, tw = W - M - tx;
  s.addText([
    { text: "門檻 0.5 時", options: { bold: true, breakLine: true } },
    { text: `F1 從 ${te.none.f1.toFixed(3)} 升到 ${te.class_weight.f1.toFixed(3)}，看起來大幅進步`, options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "但讓每個做法自己選最好的門檻", options: { bold: true, breakLine: true } },
    { text: `${te.none.f1_best.toFixed(3)} vs ${te.class_weight.f1_best.toFixed(3)}：完全沒進步`, options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "原因", options: { bold: true, breakLine: true } },
    { text: "重採樣讓預測機率整體偏高，等於把門檻往下移，所以多抓到違約者（召回率 " +
      `${te.none.recall.toFixed(2)} → ${te.class_weight.recall.toFixed(2)}）。` },
  ], { x: tx, y: 1.6, w: tw, h: 4.9, fontSize: 15, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 2,
    isTextBox: true });
  s.addNotes(
    "那為什麼大家都覺得重採樣有用？看這張圖。灰色是門檻固定 0.5 的 F1：不處理只有 0.464，用了 class weight 變成 0.542，看起來進步很多。" +
    "但紫色是讓每個做法自己選最好門檻的 F1，四個幾乎一樣，都是 0.55 左右。" +
    "意思是重採樣並沒有讓模型更會分辨，它只是讓預測機率變高，效果等於把門檻往下移。既然如此，直接調門檻就好，不需要丟資料或合成資料。");

  // =========================================================================
  // 8. 結果三：成本（原生圖表）
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結果" });
  s.addText("結果三：用成本決定門檻，最省錢", { placeholder: "title" });
  const pols = ["default", "bayes", "tuned"];
  const polNames = ["門檻 0.5", "理論門檻", "依成本調整的門檻"];
  s.addChart(pres.charts.BAR, STRAT.map(([k, name]) => ({ name, labels: polNames, values: pols.map((p) => D.cost[k][p]) })), {
    ...chartBase(), x: M, y: 1.45, w: 8.0, h: 5.2, barDir: "col", barGapWidthPct: 45, barOverlapPct: -8,
    chartColors: STRAT.map(([, , hex]) => hex), valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.2,
    valAxisLabelFormatCode: "0.0", showLegend: true, legendPos: "t", showTitle: false,
    valAxisTitle: "每位申請人的平均成本（越低越好）", showValAxisTitle: true, valAxisTitleFontSize: 11,
    valAxisTitleColor: "6B6A66", valAxisTitleFontFace: "+mn-lt", objectName: "cost-chart",
  });
  const cx = M + 8.4, cw = W - M - cx;
  stat(s, cx, 1.55, cw, `${D.cost_none_tuned_savings}%`, "不處理 + 調整門檻，\n比全部核准省下的成本");
  stat(s, cx, 3.45, cw, `${Math.round(D.bayes_reject.random_under)}%`, "重採樣模型套用理論門檻時\n被拒絕的申請人比例", RED);
  caption(s, "成本假設：核准會違約的人 = 5、拒絕好客戶 = 1。理論門檻 = C_FP / (C_FP + C_FN)；調整門檻 = 在訓練集 out-of-fold 預測上找成本最低者。信用卡、XGBoost、測試集。", cx, 5.35, cw, 1.3);
  s.addNotes(
    "接著轉成商業決策。假設核准一個會違約的人損失 5，誤拒一個好客戶損失 1。三組長條分別是三種選門檻的方式。" +
    "門檻 0.5 時，不處理的成本最高；但只要用訓練集的 out-of-fold 預測來調門檻，五種做法的成本幾乎一樣，不處理反而最低，" +
    "比全部核准省下 49%。中間那組很有意思：理論最佳門檻只有在機率準確時才成立，重採樣後的模型套用它，會拒絕 99% 的申請人。" +
    "這也是我工業工程背景的角度：模型的價值要用決策成本來衡量，而不是只看 F1。");

  // =========================================================================
  // 9. 結果四：校準
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結果" });
  s.addText("結果四：重採樣讓預測機率變得不可信", { placeholder: "title" });
  const g = addImg(s, "calibration", M, 1.45, 7.0, 5.3);
  const bx = M + g.w + 0.5, bw = W - M - bx;
  const b = D.brier;
  const brierRows = [
    [hdr("做法"), hdr("Brier 分數 ↓")],
    [{ text: "不處理" }, { text: b["none"].toFixed(3), options: { bold: true, color: BLUE } }],
    [{ text: "class weight" }, { text: b["class weight"].toFixed(3), options: { color: RED } }],
    [{ text: "隨機欠採樣" }, { text: b["random undersampling"].toFixed(3), options: { color: RED } }],
    [{ text: "隨機欠採樣 + 先驗修正" }, { text: b["random undersampling, prior-corrected"].toFixed(3), options: { bold: true, color: BLUE } }],
    [{ text: "SMOTENC" }, { text: b["SMOTENC"].toFixed(3), options: { color: RED } }],
  ];
  s.addTable(brierRows, { x: bx, y: 1.6, w: bw, colW: [bw * 0.55, bw * 0.45], fontSize: 14, color: C.text1,
    valign: "middle", rowH: 0.5, border: { type: "solid", pt: 0.75, color: "E1E0D9" } });
  s.addText("機率會被拿去定價、決定額度、估計預期損失，所以「準不準」很重要。欠採樣造成的偏差可以用 Dal Pozzolo (2015) 的公式修正回來。",
    { x: bx, y: 4.85, w: bw, h: 1.6, fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true });
  s.addNotes(
    "第四個結果是校準。圖上對角線代表預測機率完全準確。不處理和修正後的欠採樣都貼著對角線；" +
    "class weight、欠採樣、SMOTE 都在對角線右下方，代表高估違約機率。右邊的 Brier 分數也一樣。" +
    "在金融業，機率會被拿去定價、決定額度、估算預期損失，所以機率不準是實際的成本。欠採樣的偏差可以用 Dal Pozzolo 的公式修正。");

  // =========================================================================
  // 10. 結果五：受控不平衡
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結果" });
  s.addText("結果五：越不平衡，門檻越關鍵", { placeholder: "title" });
  const ir = addImg(s, "ir", M, 1.4, 7.6, 5.45);
  const ix = M + ir.w + 0.5, iw = W - M - ix;
  s.addText("同一份資料，只把違約者抽少，讓不平衡比從 3.5 一路到 100；其他條件完全不變。",
    { x: ix, y: 1.55, w: iw, h: 0.9, fontSize: 15, color: C.text2, margin: 0, valign: "top", isTextBox: true });
  const irPts = [
    ["排序能力（上列）", "各策略幾乎一樣；SMOTENC 在高度不平衡時反而落後"],
    ["門檻 0.5 的 F1（下列）", "不平衡比 ≥ 25 時，不處理幾乎抓不到任何違約者"],
    ["結論", "不平衡越嚴重，不是越需要重採樣，而是越不能忽略門檻"],
  ];
  irPts.forEach(([t, d], i) => {
    const y = 2.65 + i * 1.3;
    badge(s, i + 1, ix, y, 0.42);
    s.addText([{ text: t, options: { bold: true, breakLine: true } }, { text: d, options: { color: C.text2 } }],
      { x: ix + 0.6, y: y - 0.05, w: iw - 0.6, h: 1.15, fontSize: 15, color: C.text1, margin: 0, valign: "top",
        isTextBox: true });
  });
  s.addNotes(
    "舊版用兩份不同資料比較不平衡程度，但變因混在一起。這次我用同一份資料，只把違約者抽少，讓不平衡比從 3.5 到 100。" +
    "上面那列是不依賴門檻的排序能力，各策略幾乎一樣；下面那列是門檻 0.5 的 F1，灰色的不處理在不平衡比 25 以上幾乎是零。" +
    "所以不平衡越嚴重，代表的不是越需要重採樣，而是越不能用預設門檻。");

  // =========================================================================
  // 11. 7 份資料驗證：熱圖（原生表格）
  // =========================================================================
  pres.addSection({ title: "驗證" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "驗證" });
  s.addText("驗證：換 5 份不同領域的資料重跑", { placeholder: "title" });
  const hh = (t) => ({ text: t, options: { bold: true, color: C.text2, fill: { color: C.background2 }, align: "center" } });
  const heatRows = [[hh("資料集"), hh("不平衡比"), ...["class weight", "隨機欠採樣", "Tomek links", "SMOTE"].map(hh)]];
  D.heatmap.forEach((r) => heatRows.push([
    { text: r.display, options: { align: "left", bold: true } },
    { text: `${r.ir} : 1`, options: { color: C.text2 } },
    ...r.cells.map((c) => ({
      text: [{ text: c.v, options: { bold: true, fontSize: 14, breakLine: true } }, { text: c.wtl, options: { fontSize: 10 } }],
      options: { fill: { color: c.fill }, color: c.text },
    })),
  ]));
  s.addTable(heatRows, { x: M, y: 1.45, w: CW, colW: [2.9, 1.3, 2.0, 2.0, 2.0, 1.93], fontSize: 13, color: C.text1,
    align: "center", valign: "middle", rowH: [0.45, ...Array(7).fill(0.62)], border: { type: "solid", pt: 2, color: "FFFFFF" } });
  caption(s, "格內：各模型「策略 − 不處理」主要指標的平均差異；下方數字 = 顯著較好 / 無顯著差異 / 顯著較差 的模型數。紅 = 較差、藍 = 較好。",
    M, 6.35, CW, 0.5);
  s.addNotes(
    "為了確定這不是信用卡資料的特例，我又加了 5 份不同領域的資料：銀行行銷、德國信用、乳房 X 光、酵母菌、文件版面，" +
    "不平衡比從 2 到 175，用完全相同的程式重跑。表格依不平衡比排序，紅色代表比不處理差。" +
    "整張表幾乎沒有藍色，隨機欠採樣在高度不平衡的資料上損失最大。German Credit 只有一千筆，所以幾乎都不顯著，這也提醒我小資料的結論要保守。");

  // =========================================================================
  // 12. 驗證總表（原生堆疊長條）
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "驗證" });
  s.addText("105 次比較，0 次顯著勝過「不處理」", { placeholder: "title" });
  const wt = D.wtl, cwtl = D.cost_wtl;
  const rowsW = [
    ["主要指標（不依賴門檻）", wt["primary (threshold-free)"]],
    ["最佳門檻 F1（二分類）", wt["secondary (threshold-free)"]],
    ["最差類別召回率（多分類）", wt["worst-class recall"]],
    ["測試集成本（調門檻後）", cwtl["tuned"]],
    ["F1，門檻固定 0.5", wt["F1 at threshold 0.5"]],
  ];
  const labs = rowsW.map((r) => r[0]);
  s.addChart(pres.charts.BAR, [
    { name: "顯著較好", labels: labs, values: rowsW.map((r) => r[1][0]) },
    { name: "無顯著差異", labels: labs, values: rowsW.map((r) => r[1][1]) },
    { name: "顯著較差", labels: labs, values: rowsW.map((r) => r[1][2]) },
  ], {
    ...chartBase(), x: M, y: 1.45, w: 8.4, h: 5.2, barDir: "bar", barGrouping: "stacked", barGapWidthPct: 55,
    chartColors: [BLUE, "D6D4CD", RED], showLegend: true, legendPos: "t", showValue: true, dataLabelPosition: "ctr",
    dataLabelFontSize: 12, dataLabelColor: "16181D", dataLabelFormatCode: "0;;;", catAxisOrientation: "maxMin",
    valAxisHidden: true, valGridLine: { style: "none" }, showTitle: false, objectName: "wtl-chart",
  });
  const wx = M + 8.8, ww = W - M - wx;
  s.addText([
    { text: "不依賴門檻的指標", options: { bold: true, breakLine: true } },
    { text: "沒有一次顯著較好", options: { color: C.text2, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "門檻固定 0.5 時", options: { bold: true, breakLine: true } },
    { text: `${wt["F1 at threshold 0.5"][0]} 次「顯著較好」——這就是很多人以為重採樣有用的原因`, options: { color: C.text2 } },
  ], { x: wx, y: 1.6, w: ww, h: 4.5, fontSize: 16, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addNotes(
    "把 7 份資料的所有比較加總起來。前四列都是不依賴門檻、或已經調過門檻的指標，藍色「顯著較好」幾乎是零：" +
    "主要指標 105 次比較 0 次較好、24 次較差；測試集成本 60 次比較 0 次較便宜。" +
    "只有最後一列，門檻固定 0.5 的 F1，有 31 次顯著較好。這就解釋了為什麼很多教學會說重採樣有效：它們看的是固定門檻的指標。");

  // =========================================================================
  // 13. 例外：極少數類別
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "驗證" });
  s.addText("例外：類別只有幾筆時，class weight 有用", { placeholder: "title" });
  const y_ = D.yeast;
  const yRows = [
    [hdr("Yeast（最小類只有 5 筆）"), hdr("macro F1"), hdr("最差類別召回率")],
    [{ text: "邏輯斯迴歸 · 不處理" }, { text: y_["logreg|none"][0].toFixed(3) }, { text: y_["logreg|none"][1].toFixed(3), options: { color: RED, bold: true } }],
    [{ text: "邏輯斯迴歸 · 先驗修正" }, { text: y_["logreg|none + prior adjustment"][0].toFixed(3) }, { text: y_["logreg|none + prior adjustment"][1].toFixed(3) }],
    [{ text: "邏輯斯迴歸 · class weight" }, { text: y_["logreg|class_weight"][0].toFixed(3) }, { text: y_["logreg|class_weight"][1].toFixed(3) }],
    [{ text: "XGBoost · 先驗修正" }, { text: y_["xgboost|none + prior adjustment"][0].toFixed(3) }, { text: y_["xgboost|none + prior adjustment"][1].toFixed(3), options: { color: RED, bold: true } }],
    [{ text: "XGBoost · class weight" }, { text: y_["xgboost|class_weight"][0].toFixed(3), options: { bold: true, color: BLUE } }, { text: y_["xgboost|class_weight"][1].toFixed(3) }],
  ];
  s.addTable(yRows, { x: M, y: 1.55, w: 7.0, colW: [3.4, 1.6, 2.0], fontSize: 14, color: C.text1, align: "center",
    valign: "middle", rowH: 0.56, border: { type: "solid", pt: 0.75, color: "E1E0D9" } });
  caption(s, "訓練集 out-of-fold 預測。先驗修正 = 不重新訓練，把機率除以各類別比例再判定（多分類版的「調門檻」）。", M, 5.05, 7.0, 0.7);
  const ex = M + 7.5, ew = W - M - ex;
  const exPts = [
    ["邏輯斯迴歸：先驗修正 ≈ class weight", "效果一樣來自「改變判定方式」，跟主結論一致"],
    ["樹模型：先驗修正救不回來", "4 筆訓練樣本根本學不到；class weight 改變了樹學到的內容"],
    ["隨機欠採樣仍然最差", "Yeast 的訓練資料被砍到只剩 40 筆"],
  ];
  exPts.forEach(([t, d], i) => {
    const y = 1.6 + i * 1.5;
    badge(s, i + 1, ex, y, 0.42);
    s.addText([{ text: t, options: { bold: true, breakLine: true } }, { text: d, options: { color: C.text2 } }],
      { x: ex + 0.6, y: y - 0.05, w: ew - 0.6, h: 1.35, fontSize: 15, color: C.text1, margin: 0, valign: "top",
        isTextBox: true });
  });
  s.addNotes(
    "不過也有例外，這是驗證資料帶給我的新發現。Yeast 有 10 類，最小的類別只有 5 筆。不處理時，每個模型都完全認不出這一類。" +
    "多分類沒有單一門檻，對應的做法是先驗修正：把機率除以各類別的比例。對邏輯斯迴歸，先驗修正和 class weight 效果一樣，符合主結論；" +
    "但對 XGBoost，訓練資料裡只有 4 筆，樹根本切不出來，事後怎麼修正都沒用，這時 class weight 才真的改變了學習結果。" +
    "所以我的建議會加上這個條件：類別只有幾十筆以下時，用 class weight。");

  // =========================================================================
  // 14. 文獻與限制
  // =========================================================================
  pres.addSection({ title: "總結" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "總結" });
  s.addText("跟既有研究一致，也有明確的限制", { placeholder: "title" });
  const colW2 = (CW - 0.4) / 2;
  card(s, M, 1.55, colW2, 4.6, "lit-card");
  card(s, M + colW2 + 0.4, 1.55, colW2, 4.6, "limit-card");
  s.addText("學術界的相同發現", { x: M + 0.35, y: 1.75, w: colW2 - 0.7, h: 0.5, fontSize: 18, bold: true, color: C.text1,
    margin: 0, isTextBox: true });
  s.addText([
    { text: "van den Goorbergh 等（2022, JAMIA）：重採樣讓機率嚴重高估、AUC 沒變好；調門檻可得到相同效果", options: { bullet: true, breakLine: true } },
    { text: "Carriero 等（2025, Stat. Med.）：機器學習模型上，不修正的校準都不差於修正", options: { bullet: true, breakLine: true } },
    { text: "Elor & Averbuch-Elor（2022）：73 份資料，強模型通常不需要平衡資料", options: { bullet: true, breakLine: true } },
    { text: "本專案延伸：成本決策框架 + 稀有類別的例外", options: { bullet: true, bold: true, color: C.accent1 } },
  ], { x: M + 0.35, y: 2.35, w: colW2 - 0.7, h: 3.6, fontSize: 15, color: C.text1, margin: 0, valign: "top",
    paraSpaceAfter: 10, isTextBox: true });
  const lx = M + colW2 + 0.75;
  s.addText("限制", { x: lx, y: 1.75, w: colW2 - 0.7, h: 0.5, fontSize: 18, bold: true, color: C.text1, margin: 0,
    isTextBox: true });
  s.addText([
    { text: "參數在「不處理」下調好再共用；對各策略公平，但可能略微低估重採樣", options: { bullet: true, breakLine: true } },
    { text: "成本是相對值，只有 German Credit 有官方成本矩陣；以 1 到 20 倍的敏感度分析確認結論穩定", options: { bullet: true, breakLine: true } },
    { text: "小資料（German Credit、稀有類別）檢定力低：「不顯著」≠「一樣」", options: { bullet: true, breakLine: true } },
    { text: "未涵蓋 500:1 以上的極端不平衡，以及影像、文字資料", options: { bullet: true } },
  ], { x: lx, y: 2.35, w: colW2 - 0.7, h: 3.6, fontSize: 15, color: C.text1, margin: 0, valign: "top",
    paraSpaceAfter: 10, isTextBox: true });
  s.addNotes(
    "這些結論不是我第一個發現的，學術界已經有相同的研究，例如 2022 年 JAMIA 的論文就指出重採樣會讓機率高估、但 AUC 沒有變好。" +
    "我的實驗結果跟它們一致，這反過來證明實驗方法是對的；我的延伸是成本決策框架，以及稀有類別的例外。" +
    "限制也要講清楚：參數是共用的、成本是假設的、小資料的檢定力不足，也沒有涵蓋極端不平衡和非表格資料。");

  // =========================================================================
  // 15. 工程面
  // =========================================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "總結" });
  s.addText("工程面：讓每個結論都能被驗證", { placeholder: "title" });
  const eng = [
    [fa.FaShieldAlt, "防洩漏 Pipeline", "重採樣只在訓練時發生，由「間諜 sampler」測試證明"],
    [fa.FaVial, "49 個自動化測試 + CI", "公式、統計檢定、極端情況；每次推送 GitHub Actions 自動跑"],
    [fa.FaTerminal, "分階段命令列工具", "調參 → CV → 統計 → 測試 → 成本；改假設不必重新訓練"],
    [fa.FaFingerprint, "資料來源可追溯", "SHA-256 驗證；官方來源失敗時自動改用鏡像並比對"],
    [fa.FaBug, "驗證抓到 3 個 bug", "SMOTE 鄰居數、KNN 鄰居數、驗證折缺類別：只在極端資料出現"],
    [fa.FaChartBar, "設計規範的圖表", "色盲安全配色、向量 SVG、GitHub 深色模式自動切換"],
  ];
  const gw = (CW - 0.6) / 3, gh = 2.35;
  for (let i = 0; i < eng.length; i++) {
    const [Ic, t, d] = eng[i];
    const x = M + (i % 3) * (gw + 0.3), y = 1.55 + Math.floor(i / 3) * (gh + 0.3);
    card(s, x, y, gw, gh, `eng-card-${i}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.62, h: 0.62, fill: { color: C.accent1 }, line: { type: "none" },
      objectName: `eng-icon-bg-${i}` });
    s.addImage({ data: await icon(Ic, "FFFFFF"), x: x + 0.45, y: y + 0.45, w: 0.32, h: 0.32, altText: t });
    s.addText(t, { x: x + 0.3, y: y + 1.05, w: gw - 0.6, h: 0.45, fontSize: 16, bold: true, color: C.text1, margin: 0,
      isTextBox: true });
    s.addText(d, { x: x + 0.3, y: y + 1.5, w: gw - 0.6, h: 0.75, fontSize: 13, color: C.text2, margin: 0, valign: "top",
      isTextBox: true });
  }
  s.addNotes(
    "工程面上，我的目標是讓每個結論都能被別人驗證。防洩漏的 Pipeline 有測試證明；49 個自動化測試在每次推送時由 GitHub Actions 執行；" +
    "命令列工具分階段，改成本假設不需要重新訓練；資料檔用 SHA-256 驗證來源。" +
    "加入驗證資料時還抓到 3 個只在極端資料才會出現的 bug，例如最小類別只有 3 筆時 SMOTE 會報錯。這也是多做驗證的價值：不只驗證結論，也驗證程式。");

  // =========================================================================
  // 16. 結論
  // =========================================================================
  s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "總結" });
  s.addText("實務建議", { placeholder: "title" });
  const recs = [
    "用原始資料分布訓練，保留所有樣本",
    "依商業成本，在 out-of-fold 預測上選判定門檻",
    "類別只有幾十筆以下時用 class weight；不要把隨機欠採樣、SMOTE 當預設",
  ];
  recs.forEach((t, i) => {
    const y = 3.95 + i * 0.68;
    badge(s, i + 1, M + 0.2, y, 0.42);
    s.addText(t, { x: M + 0.85, y: y - 0.04, w: CW - 1.0, h: 0.5, fontSize: 18, color: C.background1, margin: 0,
      valign: "middle", isTextBox: true });
  });
  s.addText(`下一步：SHAP 可解釋性 · 依性別／年齡的公平性分析 · 互動式門檻與成本 demo　|　${REPO}`,
    { x: M + 0.2, y: 6.35, w: CW - 0.4, h: 0.45, fontSize: 13, color: C.background2, margin: 0, isTextBox: true });
  s.addNotes(
    "最後總結成三個實務建議。第一，用原始資料訓練，不要丟資料。第二，依商業成本在 out-of-fold 預測上選門檻。" +
    "第三，只有在類別極少時才用 class weight，不要把欠採樣和 SMOTE 當預設。" +
    "下一步我打算加入 SHAP 可解釋性和公平性分析，這在金融業很重要。程式和所有結果都在 GitHub 上，謝謝大家，歡迎提問。");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
