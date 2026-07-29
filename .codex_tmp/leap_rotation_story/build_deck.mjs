import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "D:/UniLab/LEAP_Ball_Rotation_Research_Story.pptx";
const TMP = "D:/UniLab/.codex_tmp/leap_rotation_story";
const W = 1280;
const H = 720;
const C = {
  ink: "#111827",
  muted: "#5B6472",
  light: "#F3F4F6",
  line: "#D1D5DB",
  blue: "#2563EB",
  blueLight: "#DBEAFE",
  cyan: "#0E7490",
  red: "#B42318",
  redLight: "#FEE4E2",
  green: "#047857",
  greenLight: "#D1FAE5",
  white: "#FFFFFF",
};
const FONT = "Microsoft JhengHei";

function box(slide, x, y, w, h, fill = C.light, line = "none", radius = "roundRect") {
  return slide.shapes.add({
    geometry: radius,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: line === "none" ? { style: "solid", fill: "none", width: 0 } : { style: "solid", fill: line, width: 1 },
  });
}

function text(slide, value, x, y, w, h, size = 22, color = C.ink, bold = false, align = "left") {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    fontSize: size,
    typeface: FONT,
    color,
    bold,
    alignment: align,
    verticalAlignment: "top",
    autoFit: "shrinkText",
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return shape;
}

function title(slide, value, n) {
  text(slide, value, 54, 36, 1160, 70, 38, C.ink, true);
  text(slide, String(n).padStart(2, "0"), 1180, 665, 44, 22, 13, C.muted, false, "right");
}

function notes(slide, sources, presenter = "") {
  slide.speakerNotes.textFrame.setText(`${presenter}\n\n[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}`.trim());
  slide.speakerNotes.setVisible(true);
}

function metric(slide, x, y, w, value, label, tone = C.blue) {
  box(slide, x, y, w, 158, C.light);
  text(slide, value, x + 22, y + 24, w - 44, 58, 38, tone, true);
  text(slide, label, x + 22, y + 94, w - 44, 44, 18, C.muted);
}

function runLabel(slide, runId, x, y, w = 500) {
  text(slide, `RUN  ${runId}`, x, y, w, 28, 16, C.blue, true);
}

function rewardRow(slide, label, value, x, y, max = 6) {
  text(slide, label, x, y, 210, 25, 17, C.ink);
  const zero = x + 288;
  const magnitude = Math.min(Math.abs(value) / max, 1) * 210;
  slide.shapes.add({ geometry: "straightConnector1", position: { left: zero, top: y + 12, width: 0, height: 18 }, line: { style: "solid", fill: C.ink, width: 1 } });
  box(slide, value >= 0 ? zero : zero - magnitude, y + 5, Math.max(magnitude, 2), 15, value >= 0 ? C.blue : C.red, "none", "rect");
  text(slide, `${value > 0 ? "+" : ""}${value}`, zero + 228, y, 70, 25, 16, value >= 0 ? C.blue : C.red, true);
}

function addFlowNode(slide, x, y, w, h, heading, body, fill) {
  box(slide, x, y, w, h, fill);
  text(slide, heading, x + 20, y + 18, w - 40, 35, 22, C.ink, true);
  text(slide, body, x + 20, y + 62, w - 40, h - 78, 17, C.muted);
}

async function main() {
  await fs.mkdir(`${TMP}/rendered`, { recursive: true });
  const deck = Presentation.create({ slideSize: { width: W, height: H } });

  // 1 — title
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    text(s, "LEAP HAND", 54, 50, 300, 32, 18, C.blue, true);
    text(s, "從持球到持續轉球", 54, 200, 900, 90, 58, C.ink, true);
    text(s, "四個任務如何逐步定位 reward、旋轉與 finger gaiting 的真正瓶頸", 58, 320, 900, 75, 25, C.muted);
    box(s, 930, 178, 260, 260, C.blueLight);
    text(s, "4", 982, 210, 150, 100, 76, C.blue, true, "center");
    text(s, "個研究任務\n一條證據鏈", 978, 320, 165, 72, 22, C.ink, true, "center");
    text(s, "UniLab · PPO · MuJoCo · Cache-based Dexterous Manipulation", 58, 620, 900, 30, 17, C.muted);
    notes(s, ["D:/UniLab/DEVELOPMENT_LOG.md", "D:/UniLab/logs/rsl_rl_ppo/*/run_config.json"], "本簡報以保存的 run config 與訓練摘要重建研究決策鏈。");
  }

  // 2 — research map
  {
    const s = deck.slides.add(); title(s, "每一個新任務，都在回答上一個任務留下的問題", 2);
    const y = 225, w = 244, h = 215, xs = [54, 360, 666, 972];
    for (let i = 0; i < 3; i++) s.shapes.add({ geometry: "straightConnector1", position: { left: xs[i] + w, top: y + 105, width: xs[i + 1] - xs[i] - w, height: 0 }, line: { style: "solid", fill: C.blue, width: 3, endArrowType: "triangle" } });
    addFlowNode(s, xs[0], y, w, h, "Sustained Cache", "多種 cache 抓姿下，能否持續旋轉？", C.blueLight);
    addFlowNode(s, xs[1], y, w, h, "Finger Gaiting", "停止旋轉，是否因為手指不會換位？", C.light);
    addFlowNode(s, xs[2], y, w, h, "Direct Rotation", "真正瓶頸是否是正向旋轉本身？", C.redLight);
    addFlowNode(s, xs[3], y, w, h, "Cache Gaiting", "先取得旋轉，再加入受約束 handoff。", C.greenLight);
    text(s, "7/21", xs[0], 470, w, 25, 16, C.muted, true, "center");
    text(s, "7/22 12:15", xs[1], 470, w, 25, 16, C.muted, true, "center");
    text(s, "7/22 21:27", xs[2], 470, w, 25, 16, C.muted, true, "center");
    text(s, "7/27", xs[3], 470, w, 25, 16, C.muted, true, "center");
    text(s, "這不是版本編號的直線升級，而是一系列可被 RunId 驗證的研究假設。", 54, 560, 1120, 45, 24, C.ink, true);
    notes(s, ["D:/UniLab/DEVELOPMENT_LOG.md:2244", "D:/UniLab/DEVELOPMENT_LOG.md:2385", "D:/UniLab/DEVELOPMENT_LOG.md:2620"], "先說明研究分支，而不是宣稱每個後續任務都更成功。");
  }

  // 3 — sustained baseline
  {
    const s = deck.slides.add(); title(s, "Sustained Cache 證明 cache reset 可行，但旋轉會中斷", 3);
    runLabel(s, "2026-07-21_18-22-39_mujoco", 54, 118, 600);
    text(s, "初版 reward", 54, 170, 300, 30, 22, C.ink, true);
    rewardRow(s, "spin progress", 1.5, 54, 220);
    rewardRow(s, "retention", 0.5, 54, 260);
    rewardRow(s, "fingertip support", 0.25, 54, 300);
    rewardRow(s, "failure", -5, 54, 340);
    metric(s, 685, 190, 250, "32.77M", "environment steps");
    metric(s, 960, 190, 250, "481.64", "mean episode steps", C.cyan);
    box(s, 685, 380, 525, 155, C.redLight);
    text(s, "觀察到的失敗模式", 710, 405, 470, 30, 22, C.red, true);
    text(s, "短暫旋轉 → 接觸不更新 → 速度下降 → 停止", 710, 455, 470, 52, 21, C.ink, true);
    text(s, "結論：cache 本身不是主要障礙；持續推動與接觸更新才是。", 54, 585, 1120, 40, 24, C.ink, true);
    notes(s, ["D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-21_18-22-39_mujoco/run_config.json", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-21_18-22-39_mujoco/run_summary.json", "D:/UniLab/DEVELOPMENT_LOG.md:2389"], "此 run 完成 999 iterations，適合作為主要 baseline，而不是 smoke run。");
  }

  // 4 — retention comparison
  {
    const s = deck.slides.add(); title(s, "Anchor shaping 修補 retention 灰色區，卻沒有解決旋轉中斷", 4);
    runLabel(s, "BASE  2026-07-21_18-22-39_mujoco", 54, 118, 540);
    runLabel(s, "TEST  2026-07-21_22-58-39_mujoco", 650, 118, 560);
    s.charts.add("bar", {
      position: { left: 54, top: 180, width: 650, height: 370 },
      categories: ["Final reward", "Best reward"],
      series: [
        { name: "Baseline", values: [5.152, 7.049], fill: "#94A3B8" },
        { name: "+ anchor proximity", values: [7.554, 8.965], fill: C.blue },
      ],
      hasLegend: true,
      legend: { position: "bottom", overlay: false },
      dataLabels: { showValue: true },
      yAxis: { max: 10, majorUnit: 2, majorGridlines: { style: "solid", fill: C.line, width: 1 } },
      chartFill: C.white,
      chartLine: { style: "solid", fill: C.white, width: 0 },
    });
    box(s, 760, 190, 430, 315, C.blueLight);
    text(s, "新增", 790, 220, 100, 25, 18, C.blue, true);
    text(s, "anchor_proximity  +0.1", 790, 260, 350, 35, 25, C.ink, true);
    text(s, "Final reward\n5.152 → 7.554\n\nEpisode steps\n481.64 → 499.19", 790, 325, 350, 140, 22, C.ink);
    text(s, "位置維持改善 ≠ 持續旋轉成功", 54, 585, 1120, 42, 27, C.red, true);
    notes(s, ["D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-21_18-22-39_mujoco/run_summary.json", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-21_22-58-39_mujoco/run_summary.json", "D:/UniLab/DEVELOPMENT_LOG.md:2297"], "兩個 run 都是 999 iterations／32.77M steps，可作較公平的單變量對照；reward 數值只在同一機制內比較。");
  }

  // 5 — finger gaiting
  {
    const s = deck.slides.add(); title(s, "Finger Gaiting 否定了「只要會換手就能持續旋轉」", 5);
    runLabel(s, "2026-07-22_15-47-22_mujoco", 54, 118, 550);
    box(s, 54, 180, 490, 330, C.light);
    text(s, "原始假設", 80, 210, 180, 30, 22, C.blue, true);
    text(s, "一根手指釋放\n→ 其他手指支撐\n→ 重新接觸\n→ 繼續推球", 80, 265, 390, 170, 28, C.ink, true);
    metric(s, 590, 180, 285, "22.33 s", "平均存活時間", C.green);
    metric(s, 900, 180, 285, "1.97%", "正向速度 gate", C.red);
    metric(s, 590, 365, 285, "<0.11%", "qualified handoff", C.red);
    metric(s, 900, 365, 285, "0", "2s / 5s / 10s success", C.red);
    text(s, "修正後的判斷：policy 連基本正向旋轉都尚未可靠取得，強制 handoff 太早。", 54, 570, 1130, 55, 24, C.ink, true);
    notes(s, ["D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallFingerGaitingRotation/2026-07-22_15-47-22_mujoco/run_config.json", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallFingerGaitingRotation/2026-07-22_15-47-22_mujoco/run_summary.json", "D:/UniLab/DEVELOPMENT_LOG.md:2624"], "這個 run 用來否定過早加入 handoff 的假設，而不是證明 finger gaiting 無價值。");
  }

  // 6 — direct
  {
    const s = deck.slides.add(); title(s, "Direct Rotation 將問題縮小到單一技能：正向旋轉 acquisition", 6);
    runLabel(s, "2026-07-22_21-27-39_mujoco", 54, 118, 550);
    text(s, "固定目標", 54, 176, 190, 28, 20, C.muted, true);
    text(s, "+Z · 0.30 rad/s", 54, 210, 420, 50, 34, C.blue, true);
    rewardRow(s, "stable rotation", 6, 54, 300);
    rewardRow(s, "rotation streak", 0.5, 54, 340);
    rewardRow(s, "stall", -1, 54, 380);
    rewardRow(s, "reverse", -4, 54, 420);
    box(s, 720, 180, 470, 330, C.redLight);
    text(s, "刻意移除", 750, 210, 200, 28, 21, C.red, true);
    text(s, "多階段 curriculum\n強制 handoff\nrelease-progress shaping\nretention / stage bonus", 750, 260, 390, 185, 25, C.ink, true);
    text(s, "99 iterations · 3.28M steps · 用途是隔離變因，不是宣告最終成功。", 54, 570, 1130, 48, 23, C.ink, true);
    notes(s, ["D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallDirectRotation/2026-07-22_21-27-39_mujoco/run_config.json", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallDirectRotation/2026-07-22_21-27-39_mujoco/run_summary.json", "D:/UniLab/DEVELOPMENT_LOG.md:2620"], "Direct reward 與其他 task 尺度不同，因此不能直接用 mean reward 排名。");
  }

  // 7 — V3A
  {
    const s = deck.slides.add(); title(s, "V3-A 把 reward 從複雜 shaping 收斂成任務本身", 7);
    runLabel(s, "2026-07-27_00-36-25_mujoco", 54, 118, 550);
    const labels = ["目標軸旋轉", "物體平移", "Anchor 誤差", "停止旋轉", "Failure"];
    const vals = [1.25, -0.3, -6, -0.05, -1];
    for (let i = 0; i < labels.length; i++) rewardRow(s, labels[i], vals[i], 54, 190 + i * 52);
    box(s, 720, 180, 470, 285, C.blueLight);
    text(s, "設計原則", 750, 210, 180, 30, 22, C.blue, true);
    text(s, "旋轉本身給 reward\n偏離 anchor 持續付成本\n停轉持續付成本\n離開 workspace 才終止", 750, 265, 390, 160, 24, C.ink, true);
    metric(s, 720, 500, 220, "13.11M", "environment steps");
    metric(s, 970, 500, 220, "495.37", "mean episode steps", C.cyan);
    notes(s, ["D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-27_00-36-25_mujoco/run_config.json", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-27_00-36-25_mujoco/run_summary.json"], "V3-A 是研究方向的簡化，不應單靠負的 total reward 判定失敗；需搭配 rotation diagnostics 和 viewer。");
  }

  // 8 — V3B cache gaiting
  {
    const s = deck.slides.add(); title(s, "V3-B 重新加入 handoff，但不再讓換手支配 reward", 8);
    runLabel(s, "2026-07-27_12-26-57_mujoco", 54, 118, 550);
    const xs = [54, 350, 646, 942];
    const heads = ["Rotate", "安全支撐", "有效 Handoff", "Stage gate"];
    const bodies = ["+Z 旋轉仍是主要 reward", "拇指保持；其他至少兩指支撐", "0.03 rad 真實角位移\n一次性 +0.05", "0 → 0.30 rad/s\n逐階增加 handoff"];
    const fills = [C.blueLight, C.light, C.greenLight, C.light];
    for (let i = 0; i < 4; i++) addFlowNode(s, xs[i], 205, 250, 240, heads[i], bodies[i], fills[i]);
    text(s, "禁止拇指 release", 54, 500, 260, 35, 22, C.red, true);
    text(s, "移除 stable-support 與 release-progress dense reward", 350, 500, 590, 35, 22, C.red, true);
    metric(s, 54, 530, 250, "19.66M", "environment steps");
    metric(s, 340, 530, 250, "600", "mean episode steps", C.green);
    text(s, "完整存活不等於成功旋轉；下一步仍須用 net angle、axis speed 與 handoff 指標判斷。", 635, 548, 555, 62, 20, C.ink, true);
    notes(s, ["D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallCacheGaitingRotation/2026-07-27_12-26-57_mujoco/run_config.json", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallCacheGaitingRotation/2026-07-27_12-26-57_mujoco/run_summary.json", "D:/UniLab/conf/ppo/task/leap_inhand_ball_cache_gaiting/mujoco.yaml"], "V3-B 把 handoff 從主要 shaping 降級為經驗證的事件與 stage gate。");
  }

  // 9 — evidence table
  {
    const s = deck.slides.add(); title(s, "六個主要 RunId 構成可追溯的研究證據鏈", 9);
    const rows = [
      ["Sustained baseline", "07-21 18:22:39", "999", "cache 可訓練；旋轉中斷"],
      ["Retention test", "07-21 22:58:39", "999", "位置維持改善"],
      ["Finger Gaiting", "07-22 15:47:22", "100", "正向旋轉仍是瓶頸"],
      ["Direct Rotation", "07-22 21:27:39", "99", "隔離 rotation acquisition"],
      ["Sustained V3-A", "07-27 00:36:25", "199", "簡化 reward"],
      ["Cache Gaiting V3-B", "07-27 12:26:57", "299", "受約束 handoff 整合"],
    ];
    const cols = [54, 340, 650, 790], widths = [270, 290, 120, 410];
    ["任務／角色", "RunId", "Iter.", "主要結論"].forEach((v, i) => text(s, v, cols[i], 145, widths[i], 30, 18, C.muted, true));
    for (let r = 0; r < rows.length; r++) {
      const y = 190 + r * 70;
      if (r % 2 === 0) box(s, 44, y - 8, 1180, 58, C.light, "none", "rect");
      rows[r].forEach((v, i) => text(s, v, cols[i], y, widths[i], 30, i === 1 ? 17 : 18, i === 1 ? C.blue : C.ink, i === 0));
    }
    text(s, "Smoke runs 只放附錄或驗證紀錄，不用來支持「學會了」的敘事。", 54, 625, 1120, 30, 20, C.red, true);
    notes(s, ["D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/*/run_summary.json", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallFingerGaitingRotation/*/run_summary.json", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallDirectRotation/*/run_summary.json", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallCacheGaitingRotation/*/run_summary.json"], "這一頁是之後報告與論文可重用的 canonical RunId mapping。");
  }

  // 10 — reward evolution
  {
    const s = deck.slides.add(); title(s, "Reward 演變的方向：從大量 shaping，回到可驗證的物理目標", 10);
    const x = [54, 360, 666, 972];
    const heading = ["Sustained", "Finger Gaiting", "Direct", "Cache Gaiting"];
    const body = [
      "spin + retention\n+ support\n+ stage bonus",
      "+ release progress\n+ handoff bonus\n+ handoff gates",
      "stable rotation\n- stall / reverse\n- palm / drift",
      "rotate + position\n+ small handoff event\n+ angle validation",
    ];
    for (let i = 0; i < 4; i++) {
      if (i < 3) s.shapes.add({ geometry: "straightConnector1", position: { left: x[i] + 235, top: 350, width: 70, height: 0 }, line: { style: "solid", fill: C.blue, width: 3, endArrowType: "triangle" } });
      addFlowNode(s, x[i], 225, 235, 250, heading[i], body[i], i === 3 ? C.greenLight : C.light);
    }
    text(s, "被保留的訊號", 54, 535, 220, 28, 19, C.blue, true);
    text(s, "目標軸旋轉 · anchor 安全 · 連續性 · failure", 285, 535, 870, 32, 23, C.ink, true);
    text(s, "被弱化的訊號", 54, 590, 220, 28, 19, C.red, true);
    text(s, "固定姿勢 · 單純 release · 大型 handoff bonus", 285, 590, 870, 32, 23, C.ink, true);
    notes(s, ["D:/UniLab/conf/ppo/task/leap_inhand_ball_sustained_cache/mujoco.yaml", "D:/UniLab/conf/ppo/task/leap_inhand_ball_finger_gaiting/mujoco.yaml", "D:/UniLab/conf/ppo/task/leap_inhand_ball_direct/mujoco.yaml", "D:/UniLab/conf/ppo/task/leap_inhand_ball_cache_gaiting/mujoco.yaml"], "Reward 演變不是單純增加項目，而是持續移除會形成 local optimum 的 shaping。");
  }

  // 11 — close
  {
    const s = deck.slides.add();
    text(s, "研究結論", 54, 50, 250, 34, 20, C.blue, true);
    text(s, "先學會旋轉，\n再要求換手。", 54, 175, 850, 170, 62, C.ink, true);
    text(s, "Cache 解決初始抓姿；Direct 隔離旋轉；Gaiting 應是受物理事件約束的後續技能，而不是主要 reward 捷徑。", 58, 400, 900, 95, 26, C.muted);
    box(s, 960, 175, 245, 300, C.blueLight);
    text(s, "下一步證據", 990, 210, 190, 30, 21, C.blue, true);
    text(s, "跨 seed\nnet turns\naxis-speed duration\nqualified handoff\nnative viewer", 990, 265, 190, 165, 23, C.ink, true);
    text(s, "RunId 讓每一個結論都能回到 checkpoint、metrics 與 viewer 驗證。", 58, 600, 1060, 38, 22, C.ink, true);
    notes(s, ["D:/UniLab/AGENTS.md#Dexterous-Manipulation-Training-Guidance", "D:/UniLab/DEVELOPMENT_LOG.md"], "收尾不要宣稱 Cache Gaiting 已成功；把下一步定義成可量測的驗證工作。");
  }

  const sourceNotes = `LEAP Ball Rotation Research Story\nGenerated from local UniLab evidence.\n\nPrimary sources:\n- D:/UniLab/DEVELOPMENT_LOG.md\n- D:/UniLab/AGENTS.md\n- D:/UniLab/conf/ppo/task/leap_inhand_ball_*/mujoco.yaml\n- D:/UniLab/logs/rsl_rl_ppo/*/run_config.json\n- D:/UniLab/logs/rsl_rl_ppo/*/run_summary.json\n`;
  await fs.writeFile(`${TMP}/source-notes.txt`, sourceNotes, "utf8");

  for (const [i, slide] of deck.slides.items.entries()) {
    const stem = `slide-${String(i + 1).padStart(2, "0")}`;
    const png = await deck.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(`${TMP}/rendered/${stem}.png`, new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${TMP}/rendered/${stem}.layout.json`, await layout.text(), "utf8");
  }
  const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(`${TMP}/montage.webp`, new Uint8Array(await montage.arrayBuffer()));
  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(OUT);
  const inspection = await deck.inspect({ kind: "slide,textbox,shape,chart", maxChars: 12000 });
  await fs.writeFile(`${TMP}/inspection.ndjson`, inspection.ndjson, "utf8");
  console.log(JSON.stringify({ output: OUT, slides: deck.slides.items.length }));
}

await main();
