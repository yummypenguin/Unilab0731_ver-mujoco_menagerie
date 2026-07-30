import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const SOURCE = "D:/UniLab/LEAP_Hand_Monthly_Progress_Report_English_14pt_CacheFlow_Performance_Updated.pptx";
const OUT = "D:/UniLab/LEAP_Hand_Monthly_Progress_Report_English_14pt_CacheFlow_FingerGaiting_Updated.pptx";
const TMP = "D:/UniLab/.codex_tmp/leap_monthly_progress_update/finger_gaiting_update";
const FONT = "Aptos";
const C = {
  ink: "#111827", muted: "#5B6472", line: "#D1D5DB", light: "#F3F4F6",
  white: "#FFFFFF", blue: "#2563EB", blueLight: "#DBEAFE", cyan: "#0E7490",
  cyanLight: "#CFFAFE", green: "#047857", greenLight: "#D1FAE5",
  amber: "#B45309", amberLight: "#FEF3C7", red: "#B42318", redLight: "#FEE4E2",
};

function box(slide, x, y, w, h, fill = C.light, line = "none", geometry = "rect") {
  return slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: line === "none" ? { style: "solid", fill: "none", width: 0 } : { style: "solid", fill: line, width: 1 },
  });
}

function text(slide, value, x, y, w, h, size = 19, color = C.ink, bold = false, align = "left", autoFit = "shrinkText") {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    fontSize: Math.max(size, 19), typeface: FONT, color, bold, alignment: align,
    verticalAlignment: "middle", autoFit,
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return shape;
}

function title(slide, value, number) {
  text(slide, "LEAP HAND MONTHLY PROGRESS", 54, 24, 900, 30, 19, C.blue, true, "left", "none");
  text(slide, value, 54, 58, 1145, 52, 36, C.ink, true, "left", "none");
  text(slide, String(number).padStart(2, "0"), 1024, 664, 176, 30, 19, C.muted, false, "right", "none");
}

function notes(slide, sources, presenter) {
  slide.speakerNotes.textFrame.setText(`${presenter}\n\n[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}`);
  slide.speakerNotes.setVisible(true);
}

function table(slide, { x, y, widths, headers, rows, rowHeight, stageTones = [] }) {
  const hh = 38;
  let xx = x;
  headers.forEach((header, i) => {
    box(slide, xx, y, widths[i], hh, C.ink, C.white);
    text(slide, header, xx + 7, y, widths[i] - 14, hh, 19, C.white, true, i >= 3 ? "center" : "left", "none");
    xx += widths[i];
  });
  rows.forEach((row, r) => {
    let cellX = x;
    const yy = y + hh + r * rowHeight;
    row.forEach((value, c) => {
      const fill = r % 2 === 0 ? C.light : C.white;
      box(slide, cellX, yy, widths[c], rowHeight, fill, C.line);
      if (c === 0 && stageTones[r]) {
        box(slide, cellX + 7, yy + 11, widths[c] - 14, rowHeight - 22, stageTones[r], "none", "roundRect");
        text(slide, value, cellX + 7, yy + 11, widths[c] - 14, rowHeight - 22, 19, C.white, true, "center", "none");
      } else {
        const numeric = c >= 3 && c <= 6;
        text(slide, value, cellX + 7, yy + 5, widths[c] - 14, rowHeight - 10, 19, c === 1 ? C.blue : C.ink, c === 1, numeric ? "center" : "left");
      }
      cellX += widths[c];
    });
  });
}

function callout(slide, x, y, w, heading, body, fill, tone) {
  box(slide, x, y, w, 130, fill, "none", "roundRect");
  text(slide, heading, x + 18, y + 12, w - 36, 30, 19, tone, true, "left", "none");
  text(slide, body, x + 18, y + 47, w - 36, 68, 19, C.ink, false, "left", "shrinkText");
}

function buildSlide9(slide) {
  slide.shapes.deleteAll();
  slide.background.fill = C.white;
  title(slide, "Four Valid Experiments Trace the Finger-Gaiting Bottleneck", 9);
  text(slide, "Final 10% iteration mean · five run directories form four experiments · ten 1-iteration smoke runs excluded", 54, 111, 1110, 24, 19, C.muted, false, "left", "none");

  const rows = [
    ["FG1", "12-19-14 → 12-36-43", "100-observation baseline", "+.00557", "11.27", ".0081%", "16.24", "Stable hold; no 2 s spin"],
    ["FG2", "12-59-49", "+ Markov handoff state", "+.00421", "16.76", ".0057%", "12.38", "Observation alone did not help"],
    ["FG3", "14-58-36", "+ stationary handoff stage", "+.00864", "7.49", ".1290%", "13.69", "Handoffs rose; duration invalid*"],
    ["FG4", "15-47-22", "Same reward; corrected diagnostics", "−.00751", "10.59", ".0806%", "21.30", "Corrected 2/5/10 s = 0%"],
  ];
  table(slide, {
    x: 54, y: 143,
    widths: [70, 210, 250, 90, 86, 108, 90, 232],
    headers: ["Gen.", "Effective RunId", "Intervention", "Axis EMA", "Error\n(mm)", "Qualified\nhandoff", "Survival\n(s)", "Observed result"],
    rows, rowHeight: 67,
    stageTones: [C.blue, C.cyan, C.amber, C.green],
  });

  callout(slide, 54, 470, 352, "FG1 → FG2: no reward change", "Making the handoff state observable reduced speed, retention, handoffs, and survival. Sparse eligibility remained the limiting factor.", C.blueLight, C.blue);
  callout(slide, 423, 470, 352, "FG2 → FG3: handoff unlocked", "The zero-speed stage raised qualified handoffs 22.6× and total handoffs 89×; 42.6% reached the first rotation stage.", C.amberLight, C.amber);
  callout(slide, 792, 470, 354, "FG4: rotation still absent", "Correct telemetry shows long survival but negative axis EMA and zero sustained 2/5/10 s success. *FG3 duration values were false positives.", C.redLight, C.red);

  notes(slide, [
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallFingerGaitingRotation/2026-07-22_12-19-14_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallFingerGaitingRotation/2026-07-22_12-36-43_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallFingerGaitingRotation/2026-07-22_12-59-49_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallFingerGaitingRotation/2026-07-22_14-58-36_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallFingerGaitingRotation/2026-07-22_15-47-22_mujoco",
    "D:/UniLab/DEVELOPMENT_LOG.md:2385-2630",
  ], "Metrics are arithmetic means over the final 10% of TensorBoard iteration records. FG1 combines the fresh run and its checkpoint-100 resume. FG3 sustained-duration metrics are excluded because the stationary stage was incorrectly counted as rotation; FG4 contains the corrected diagnostics.");
}

function rewardRow(slide, y, label, value, tone, fill) {
  box(slide, 54, y, 520, 40, fill, C.line);
  text(slide, label, 68, y, 330, 40, 19, C.ink, false, "left", "none");
  text(slide, value, 410, y, 145, 40, 19, tone, true, "right", "none");
}

function gateRow(slide, y, label, value, tone, fill) {
  box(slide, 610, y, 536, 40, fill, C.line);
  text(slide, label, 626, y, 330, 40, 19, C.ink, false, "left", "none");
  text(slide, value, 972, y, 154, 40, 19, tone, true, "right", "none");
}

function buildSlide10(slide) {
  slide.shapes.deleteAll();
  slide.background.fill = C.white;
  title(slide, "The Reward Favors Retention Before Rotating Handoffs", 10);
  text(slide, "FG4 final 10% mean · approximate per-step contribution after ctrl_dt = 0.05 s", 54, 111, 950, 24, 19, C.muted, false, "left", "none");

  text(slide, "Reward contribution", 54, 144, 350, 30, 23, C.blue, true, "left", "none");
  text(slide, "First rotation stage: gate pass rate", 610, 144, 500, 30, 23, C.green, true, "left", "none");
  const rewards = [
    ["anchor proximity", "+0.00470", C.green], ["hold", "+0.00174", C.green],
    ["stable support", "+0.00056", C.green], ["rotation stability", "+0.00016", C.blue],
    ["handoff event", "+0.00008", C.blue], ["release progress", "+0.00003", C.blue],
    ["spin progress", "−0.00123", C.red],
  ];
  rewards.forEach((r, i) => rewardRow(slide, 180 + i * 40, r[0], r[1], r[2], i % 2 === 0 ? C.light : C.white));

  const gates = [
    ["no failure", "99.73%", C.green], ["orthogonal speed", "93.82%", C.green],
    ["fingertip support", "85.97%", C.green], ["retention", "77.76%", C.green],
    ["positive-speed gate", "1.92%", C.red], ["handoff completion", "0.88%", C.red],
    ["final stage-valid", "0.57%", C.red],
  ];
  gates.forEach((g, i) => gateRow(slide, 180 + i * 40, g[0], g[1], g[2], i % 2 === 0 ? C.light : C.white));

  const flowY = 510;
  const flow = [
    [54, "Anchor + hold\ndominate"], [284, "Policy preserves\ncache grasp"],
    [514, "Axis EMA stays\nnear zero"], [744, "Speed gate opens\nonly 1.92%"],
    [974, "No rotating handoff\nor sustained spin"],
  ];
  for (let i = 0; i < flow.length - 1; i++) {
    slide.shapes.add({
      geometry: "straightConnector1",
      position: { left: flow[i][0] + 196, top: flowY + 43, width: 34, height: 0 },
      line: { style: "solid", fill: C.line, width: 3, endArrowType: "triangle" },
    });
  }
  flow.forEach((item, i) => {
    const tone = i < 2 ? C.green : i < 4 ? C.amber : C.red;
    const fill = i < 2 ? C.greenLight : i < 4 ? C.amberLight : C.redLight;
    box(slide, item[0], flowY, 196, 86, fill, "none", "roundRect");
    text(slide, item[1], item[0] + 10, flowY + 8, 176, 70, 19, tone, true, "center", "shrinkText");
  });
  text(slide, "Circular dependency: rotating handoff eligibility requires axis EMA ≥ 0.60 × 0.04 = 0.024 rad/s before the handoff bonus can bootstrap.", 54, 612, 1100, 34, 19, C.red, true, "left", "none");

  notes(slide, [
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallFingerGaitingRotation/2026-07-22_15-47-22_mujoco/events.out.tfevents.*",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallFingerGaitingRotation/2026-07-22_15-47-22_mujoco/run_config.json",
    "D:/UniLab/src/unilab/envs/manipulation/leap_inhand/finger_gaiting_rotation.py:402-547",
    "D:/UniLab/src/unilab/envs/manipulation/leap_inhand/sustained_rotation.py:173-225",
    "D:/UniLab/src/unilab/envs/manipulation/leap_inhand/sustained_rotation.py:470-642",
  ], "Dense reward terms are logged before multiplication by ctrl_dt; the displayed reward contributions multiply those terms by 0.05. The handoff event is already an event reward and is not multiplied by ctrl_dt. Level-2 gate rates use the corrected per-level diagnostics from FG4.");
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(`${TMP}/rendered`, { recursive: true });
  const deck = await PresentationFile.importPptx(await FileBlob.load(SOURCE));
  if (deck.slides.items.length !== 16) throw new Error(`Expected 16 slides, found ${deck.slides.items.length}`);
  buildSlide9(deck.slides.items[8]);
  buildSlide10(deck.slides.items[9]);

  for (const [index, slide] of deck.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(`${TMP}/rendered/${stem}.png`, await deck.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${TMP}/rendered/${stem}.layout.json`, await layout.text());
  }
  const inspect = await deck.inspect({ kind: "slide,textbox,shape,notes,layout", maxChars: 100000 });
  await fs.writeFile(`${TMP}/rendered/inspection.ndjson`, inspect.ndjson);
  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(OUT);
  console.log(`Updated ${OUT}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
