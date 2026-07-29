import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "D:/UniLab/LEAP_Hand_Monthly_Progress_Report_English_14pt.pptx";
const TMP = "D:/UniLab/.codex_tmp/leap_monthly_progress";
const EVOLUTION_IMAGE = "C:/Users/timli/Downloads/ChatGPT Image 2026年7月28日 下午05_27_48.png";
const W = 1280;
const H = 720;
const FONT = "Aptos";
const C = {
  ink: "#111827",
  muted: "#5B6472",
  line: "#D1D5DB",
  light: "#F3F4F6",
  white: "#FFFFFF",
  blue: "#2563EB",
  blueLight: "#DBEAFE",
  cyan: "#0E7490",
  cyanLight: "#CFFAFE",
  green: "#047857",
  greenLight: "#D1FAE5",
  amber: "#B45309",
  amberLight: "#FEF3C7",
  red: "#B42318",
  redLight: "#FEE4E2",
};

function box(slide, x, y, w, h, fill = C.light, line = "none", geometry = "rect") {
  return slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: line === "none"
      ? { style: "solid", fill: "none", width: 0 }
      : { style: "solid", fill: line, width: 1 },
  });
}

function text(slide, value, x, y, w, h, size = 20, color = C.ink, bold = false, align = "left", autoFit = "shrinkText") {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    // Artifact Tool uses CSS pixels; 19 px exports as 14.25 pt in PowerPoint.
    fontSize: Math.max(size, 19),
    typeface: FONT,
    color,
    bold,
    alignment: align,
    verticalAlignment: "top",
    autoFit,
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return shape;
}

function title(slide, value, number, eyebrow = "LEAP HAND MONTHLY PROGRESS") {
  text(slide, eyebrow, 54, 24, 900, 30, 14, C.blue, true, "left", "none");
  text(slide, value, 54, 58, 1135, 62, 38, C.ink, true, "left", "none");
  text(slide, String(number).padStart(2, "0"), 1024, 664, 176, 30, 19, C.muted, false, "right", "none");
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function notes(slide, sources, presenter = "") {
  slide.speakerNotes.textFrame.setText(
    `${presenter}\n\n[Sources]\n${sources.map((source) => `- ${source}`).join("\n")}`.trim(),
  );
  slide.speakerNotes.setVisible(true);
}

function metric(slide, x, y, w, value, label, tone = C.blue) {
  box(slide, x, y, w, 140, C.light, "none", "roundRect");
  text(slide, value, x + 20, y + 20, w - 40, 52, 36, tone, true);
  text(slide, label, x + 20, y + 82, w - 40, 40, 17, C.muted);
}

function runLabel(slide, runId, x, y, w = 500, tone = C.blue) {
  text(slide, `RUN  ${runId}`, x, y, w, 24, 15, tone, true);
}

function sectionBand(slide, y, label, tone = C.blue) {
  box(slide, 54, y, 8, 34, tone);
  text(slide, label, 78, y + 2, 1100, 32, 23, C.ink, true);
}

function timelineItem(slide, x, y, date, heading, detail, tone) {
  box(slide, x, y, 12, 12, tone, "none", "ellipse");
  text(slide, date, x - 8, y - 44, 128, 24, 15, tone, true);
  text(slide, heading, x - 8, y + 24, 186, 46, 20, C.ink, true);
  text(slide, detail, x - 8, y + 78, 190, 92, 16, C.muted);
}

function rewardTable(slide, rows, x, y, w, rowH = 42) {
  const labelW = w * 0.54;
  const valueW = w - labelW;
  rows.forEach((row, index) => {
    const yy = y + index * rowH;
    const fill = index % 2 === 0 ? C.light : C.white;
    box(slide, x, yy, labelW, rowH, fill, C.line);
    box(slide, x + labelW, yy, valueW, rowH, fill, C.line);
    text(slide, row[0], x + 12, yy + 9, labelW - 24, rowH - 14, 16, C.ink, index === 0);
    const tone = String(row[1]).startsWith("-") ? C.red : C.blue;
    text(slide, row[1], x + labelW + 12, yy + 9, valueW - 24, rowH - 14, 16, index === 0 ? C.ink : tone, true, "right");
  });
}

function phaseRow(slide, y, phase, change, runId, tone) {
  box(slide, 54, y, 105, 54, tone);
  text(slide, phase, 54, y + 13, 105, 30, 18, C.white, true, "center");
  text(slide, change, 186, y + 4, 630, 46, 17, C.ink);
  text(slide, runId, 842, y + 9, 362, 34, 15, C.blue, true);
}

async function main() {
  await fs.mkdir(`${TMP}/rendered`, { recursive: true });
  const deck = Presentation.create({ slideSize: { width: W, height: H } });

  // 1. Title
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    text(s, "UNILAB · DEXTEROUS MANIPULATION", 54, 44, 720, 26, 16, C.blue, true);
    text(s, "LEAP Hand Dexterous Manipulation\nOne-Month Research Progress", 54, 170, 890, 174, 54, C.ink, true, "left", "none");
    text(s, "An evidence chain from model integration and ball caches to sustained rotation and finger gaiting", 58, 382, 880, 70, 24, C.muted);
    box(s, 1010, 170, 180, 280, C.blueLight);
    text(s, "5", 1027, 202, 145, 94, 74, C.blue, true, "center");
    text(s, "cache tasks\n13 reward versions", 1028, 316, 145, 82, 20, C.ink, true, "center");
    text(s, "Reporting period: July 13–28, 2026", 58, 630, 620, 28, 17, C.muted);
    notes(s, [
      "D:/UniLab/DEVELOPMENT_LOG.md",
      "D:/UniLab/logs/rsl_rl_ppo/*/run_config.json",
      "D:/UniLab/logs/rsl_rl_ppo/*/run_summary.json",
    ], "This report reconstructs progress from saved code, configurations, run metadata, TensorBoard diagnostics, and native-viewer inspection.");
  }

  // 2. Executive summary
  {
    const s = deck.slides.add(); title(s, "From Trainable Environments to Diagnosable Experiments", 2);
    metric(s, 54, 170, 330, "2", "MuJoCo / Motrix backend");
    metric(s, 420, 170, 330, "50k", "LEAP-specific ball grasp states", C.green);
    metric(s, 786, 170, 330, "5", "independently comparable cache tasks", C.cyan);
    sectionBand(s, 365, "Core Deliverables");
    text(s, "Model and Physics", 78, 420, 250, 30, 22, C.ink, true);
    text(s, "Imported LEAP assets, dedicated ball/cube scenes, self-collision, and adjacent-link contact filtering.", 78, 462, 300, 100, 18, C.muted);
    text(s, "Reset Distribution", 430, 420, 250, 30, 22, C.ink, true);
    text(s, "Identified the cube-cache mismatch, built a 50k ball-specific cache, and stored a per-reset anchor.", 430, 462, 300, 100, 18, C.muted);
    text(s, "Research Method", 782, 420, 250, 30, 22, C.ink, true);
    text(s, "Used checkpoints, task diagnostics, the MuJoCo viewer, and RunIds for controlled reward and curriculum analysis.", 782, 462, 330, 100, 18, C.muted);
    text(s, "Sustained rotation is not yet claimed as solved; the ordering of acquisition, retention, and handoff is now measurable.", 54, 612, 1130, 36, 22, C.red, true);
    notes(s, ["D:/UniLab/AGENTS.md", "D:/UniLab/DEVELOPMENT_LOG.md"], "The emphasis is a reliable experimental foundation and traceable evidence, not the highest aggregate reward.");
  }

  // 3. Month timeline
  {
    const s = deck.slides.add(); title(s, "Platform First, Then the Sustained-Rotation Bottleneck", 3);
    s.shapes.add({ geometry: "straightConnector1", position: { left: 82, top: 292, width: 1090, height: 0 }, line: { style: "solid", fill: C.line, width: 3 } });
    timelineItem(s, 92, 286, "7/13–14", "Environment", "Windows 11 setup, MuJoCo smoke tests, Motrix training, and diagnosis of wgpu OOM recording failures.", C.blue);
    timelineItem(s, 315, 286, "7/14–17", "LEAP Integration", "Imported the hand and dedicated ball/cube assets; added in-hand rotation and toss tasks.", C.cyan);
    timelineItem(s, 538, 286, "7/17–21", "Physics and Cache", "Corrected penetration and self-collision behavior; produced the ball-specific 50k reset cache.", C.green);
    timelineItem(s, 761, 286, "7/21–23", "Sustained Spin", "Created sustained, retention, gaiting, and direct-rotation diagnostic branches.", C.amber);
    timelineItem(s, 984, 286, "7/25–28", "Simplification", "Removed ineffective shaping and converged on the V3-A / V3-B cache-gaiting direction.", C.red);
    text(s, "Experimental policy: integration smoke test → short diagnostic run → long training only after the direction is supported.", 54, 584, 1130, 42, 22, C.ink, true);
    notes(s, ["D:/UniLab/DEVELOPMENT_LOG.md", "D:/UniLab/AGENTS.md#Dexterous-Manipulation-Training-Guidance"]);
  }

  // 4. Backend setup
  {
    const s = deck.slides.add(); title(s, "Both Backends Train; Rendering Reliability Differs", 4);
    box(s, 54, 160, 520, 380, C.blueLight);
    text(s, "MuJoCo", 86, 190, 220, 40, 30, C.blue, true);
    text(s, "• PPO smoke and formal training run successfully\n• Native viewer supports checkpoint inspection\n• Strong XML, contact, and penetration diagnostics\n• Primary backend for current research decisions", 86, 260, 430, 220, 20, C.ink);
    box(s, 626, 160, 520, 380, C.light);
    text(s, "Motrix", 658, 190, 220, 40, 30, C.cyan, true);
    text(s, "• Training and interactive playback run successfully\n• Intel Arc recording triggered wgpu out-of-memory\n• Vulkan and DX12 both closed the render channel\n• Retained for backend-contract and performance checks", 658, 260, 430, 220, 20, C.ink);
    text(s, "Decision: use the native MuJoCo viewer for behavioral evidence; do not treat Motrix recording failures as training or physics failures.", 54, 592, 1120, 48, 21, C.ink, true);
    notes(s, ["D:/UniLab/conf/ppo/task/leap_inhand_ball_*/mujoco.yaml", "D:/UniLab/conf/ppo/task/leap_inhand_ball_*/motrix.yaml", "D:/UniLab/DEVELOPMENT_LOG.md"]);
  }

  // 5. Cache generation
  {
    const s = deck.slides.add(); title(s, "How the Production 50k Ball Cache Is Collected", 5);
    const stages = [
      ["1", "Validated seed", "Start from grasp_seed_qpos: a ball-owned state refined through penetration coordinate scans and full settling checks. Cube and toss caches are never read.", C.blue],
      ["2", "Reset proposals", "The current owner samples every hand joint within ±0.10 rad, clips to LEAP joint limits, keeps ball-position noise at zero, and initializes zero velocity.", C.cyan],
      ["3", "Zero-action hold", "apply_action() discards policy output. PD control holds each sampled reset target for a 2.5 s episode; strict validation begins after the 0.5 s warm-up.", C.amber],
      ["4", "Save only survivors", "Only non-terminated truncations are serialized to float32, reconstructed in MuJoCo, rechecked, deduplicated, and atomically published until 50,000 rows are reached.", C.green],
    ];
    stages.forEach((item, i) => {
      const y = 148 + i * 116;
      box(s, 54, y, 62, 84, item[3], "none", "roundRect");
      text(s, item[0], 54, y + 22, 62, 36, 26, C.white, true, "center");
      text(s, item[1], 142, y + 2, 300, 30, 22, C.ink, true);
      text(s, item[2], 142, y + 38, 620, 64, 17, C.muted);
    });
    box(s, 820, 148, 350, 440, C.light, C.line, "roundRect");
    text(s, "Production acceptance contract", 848, 176, 294, 58, 24, C.blue, true);
    text(s, "During the episode\n• finite state and valid joint limits\n• object remains above reset height\n• anchor drift ≤ 5 mm\n• all tips within 0.10 m\n• at least two contacts, including thumb\n• bounded ball/joint speed and work\n\nAfter float32 reconstruction\n• contact gate repeated\n• every tip surface gap ≤ 9.95 mm\n• self and object penetration ≤ 1 mm", 848, 248, 294, 300, 17, C.ink);
    text(s, "Row format: 16 hand qpos + 3 ball position + 4 quaternion = 23 float32 values", 54, 625, 1110, 32, 18, C.green, true);
    notes(s, ["D:/UniLab/src/unilab/envs/manipulation/leap_inhand/ball_grasp_gen.py", "D:/UniLab/src/unilab/envs/manipulation/allegro_inhand/grasp_gen.py", "D:/UniLab/conf/ppo/task/leap_inhand_ball_grasp/mujoco.yaml", "D:/UniLab/src/unilab/assets/robots/leap_hand/caches/ball_grasp_official_50k.npy"]);
  }

  // 6. Task evolution
  {
    const s = deck.slides.add(); title(s, "Why the Cache-Rotation Task Branched", 6);
    const imageBytes = await readImageBlob(EVOLUTION_IMAGE);
    s.images.add({
      blob: imageBytes,
      contentType: "image/png",
      alt: "Task evolution for cache rotation and gaiting",
      fit: "contain",
      position: { left: 54, top: 142, width: 830, height: 468 },
    });
    text(s, "How to read the map", 920, 150, 300, 34, 24, C.blue, true);
    text(s, "1  Sustained Cache is the common multi-grasp rotation problem.\n\n2  Finger Gaiting tests whether the first spin stalls because the hand cannot change contacts.\n\n3  V3-A removes complex shaping and returns to an Allegro-style rotation objective.\n\n4  Direct Rotation is supporting evidence: positive rotation can be acquired without a handoff objective.\n\n5  Cache Gaiting V3-B recombines the simplified rotation base with constrained handoff.", 920, 206, 300, 330, 18, C.ink);
    text(s, "Interpretation", 920, 558, 180, 28, 21, C.green, true);
    text(s, "The branches are controlled diagnostic experiments, not a linear sequence of task replacements.", 920, 596, 300, 60, 17, C.muted);
    notes(s, [EVOLUTION_IMAGE, "D:/UniLab/src/unilab/envs/manipulation/leap_inhand/sustained_cache_rotation.py", "D:/UniLab/src/unilab/envs/manipulation/leap_inhand/finger_gaiting_rotation.py", "D:/UniLab/src/unilab/envs/manipulation/leap_inhand/direct_rotation.py", "D:/UniLab/src/unilab/envs/manipulation/leap_inhand/sustained_cache_gaiting_rotation.py"]);
  }

  // 7. Sustained evolution 1
  {
    const s = deck.slides.add(); title(s, "Sustained Cache S1–S6: From Curriculum to Fixed Speed", 7);
    phaseRow(s, 132, "S1", "spin +1.5；retention +0.5；support +0.25；action -0.001；torque -0.005；work -0.05；failure -5", "2026-07-21_17-50-30", C.blue);
    phaseRow(s, 200, "S2", "+ anchor_proximity 0.1 to pull the 15–30 mm grey zone back toward the cache anchor", "2026-07-21_22-47-15", C.green);
    phaseRow(s, 268, "S3", "Level-dependent positive-spin retention floor: L0–2=0.5, L3–7=0.25", "2026-07-22_02-36-35", C.cyan);
    phaseRow(s, 336, "S4", "Disable anchor reward; restore all floors to 0.5; enable direct_spin_reward", "2026-07-22_23-04-12", C.amber);
    phaseRow(s, 404, "S5", "Remove the eight-stage curriculum; fix target at 0.30 rad/s; spin=3.0; floor=1.0", "2026-07-23_00-19-50", C.red);
    phaseRow(s, 472, "S6", "+ spin_continuity -0.05 to address the policy stopping after the initial spin", "2026-07-23_02-11-05", C.blue);
    box(s, 54, 552, 520, 98, C.blueLight, "none", "roundRect");
    text(s, "Generation 1 · S1–S3", 76, 566, 250, 24, 18, C.blue, true);
    text(s, "Hold → slow spin → faster spin. Retention and support could substitute for actual rotation.", 76, 600, 470, 42, 16, C.ink);
    box(s, 610, 552, 536, 98, C.amberLight, "none", "roundRect");
    text(s, "Generation 2 · S4–S6", 632, 566, 250, 24, 18, C.amber, true);
    text(s, "Use one fixed target speed and continuity pressure to isolate positive-rotation acquisition.", 632, 600, 480, 42, 16, C.ink);
    notes(s, ["D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/*/run_config.json", "D:/UniLab/DEVELOPMENT_LOG.md:2244", "D:/UniLab/DEVELOPMENT_LOG.md:2297"]);
  }

  // 8. Sustained evolution 2
  {
    const s = deck.slides.add(); title(s, "Sustained Cache S7–S13: Converging on V3-A", 8);
    phaseRow(s, 126, "S7", "Disable old shaping; keep Allegro rotate +1.25; failure=0; allow 10 mm drop", "2026-07-23_03-57-44", C.blue);
    phaseRow(s, 188, "S8", "+ obj_linvel -0.3 to suppress whole-object translation", "2026-07-23_15-34-09", C.cyan);
    phaseRow(s, 250, "S9", "+ position_error -5; relax workspace termination to 50 mm", "2026-07-23_20-18-51", C.green);
    phaseRow(s, 312, "S10", "Contact-weighted rotation via base / index / middle: 0.25 / 0.375 / 0.375", "2026-07-23_21-54-51", C.amber);
    phaseRow(s, 374, "S11", "+ support-pose 0.25、opposition-progress 0.20", "2026-07-25_03-50-30", C.red);
    phaseRow(s, 436, "S12", "V3-A: remove pose shaping; position -5→-6; failure 0→-1; square the gate", "2026-07-26_23-27-56", C.blue);
    phaseRow(s, 498, "S13", "Current version: V3-A + position-gated spin_continuity -0.05", "2026-07-27_00-36-25", C.green);
    text(s, "Current formula", 54, 570, 150, 24, 18, C.green, true);
    text(s, "e=‖ball-anchor‖；p=clip(axis_speed/0.30,-1,1)；purity=exp(-(orth_speed/0.10)²)；gate=position_gate(e)²", 180, 570, 1020, 26, 16, C.muted);
    text(s, "rotate: forward=1.25×0.30×clip(p,0,1)×purity×gate; reverse=1.25×0.30×clip(p,-1,0)", 210, 602, 990, 26, 16, C.ink, true);
    text(s, "reward=dt×(rotate-0.3‖v‖₁-6e-0.05×gate×(1-clip(p,0,1)))-failure; gate=1 within 15 mm; terminate beyond 50 mm", 210, 634, 990, 26, 16, C.ink);
    notes(s, ["D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/*/run_config.json", "D:/UniLab/conf/ppo/task/leap_inhand_ball_sustained_cache/mujoco.yaml"]);
  }

  // 9. Grey zone evidence
  {
    const s = deck.slides.add(); title(s, "Anchor Proximity Improved Retention, Not Sustained Rotation", 9);
    runLabel(s, "Baseline  2026-07-21_18-22-39_mujoco", 54, 138, 520);
    runLabel(s, "Grey-zone  2026-07-21_22-58-39_mujoco", 650, 138, 520, C.green);
    const cols = [54, 350, 550, 750];
    const widths = [296, 200, 200, 430];
    const rows = [
      ["Diagnostic", "Baseline", "+ anchor", "Interpretation"],
      ["anchor position error", "23.35 mm", "20.09 mm", "14% lower; object stays closer to reset anchor"],
      ["fingertip contacts", "2.15", "3.00", "Stronger enclosure and retention"],
      ["stage-valid fraction", "10.96%", "16.22%", "+5.26 percentage points of valid states"],
      ["axis-speed EMA", "+0.010", "+0.013 rad/s", "Only a small gain; still far below target speed"],
      ["sustained ≥2 s", "0.45%", "2.16%", "Short successes increased but remained rare"],
      ["sustained ≥5 / 10 s", "0% / 0%", "0% / 0%", "No sustained rotation was acquired"],
      ["survival seconds", "13.11 s", "12.78 s", "No survival improvement"],
    ];
    rows.forEach((row, ri) => row.forEach((value, ci) => {
      const yy = 190 + ri * 50;
      box(s, cols[ci], yy, widths[ci], 50, ri === 0 ? C.ink : (ri % 2 ? C.white : C.light), C.line);
      text(s, value, cols[ci] + 10, yy + 10, widths[ci] - 20, 30, ri === 0 ? 16 : 15, ri === 0 ? C.white : (ci === 2 ? C.green : C.ink), ri === 0 || ci === 2);
    }));
    text(s, "Conclusion: anchor proximity repaired the 15–30 mm grey zone and improved contact retention, but it did not create long-duration positive angular speed or finger switching.", 54, 610, 1130, 48, 20, C.red, true);
    notes(s, ["D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-21_18-22-39_mujoco/events.out.tfevents.1784683360.沅霆.15996.0", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-21_22-58-39_mujoco/events.out.tfevents.1784699921.沅霆.13112.0", "D:/UniLab/DEVELOPMENT_LOG.md:2297"]);
  }

  // 12. Finger gaiting design
  {
    const s = deck.slides.add(); title(s, "Finger Gaiting: Testing Contact Switching", 10);
    text(s, "Observed failure sequence", 54, 150, 360, 30, 22, C.ink, true);
    const seq = ["Initial push", "Brief +Z spin", "Contacts freeze", "Speed decays to zero"];
    seq.forEach((label, i) => {
      if (i < seq.length - 1) s.shapes.add({ geometry: "straightConnector1", position: { left: 170 + i * 275, top: 245, width: 82, height: 0 }, line: { style: "solid", fill: C.red, width: 3, endArrowType: "triangle" } });
      box(s, 54 + i * 275, 205, 195, 80, i === 3 ? C.redLight : C.light, "none", "roundRect");
      text(s, label, 70 + i * 275, 229, 163, 30, 19, i === 3 ? C.red : C.ink, true, "center");
    });
    sectionBand(s, 350, "Added rewards and gates", C.amber);
    rewardTable(s, [["Gaiting term", "Weight"], ["stable_support", "+0.20"], ["release_progress", "+0.05"], ["qualified_handoff", "+0.25"]], 54, 408, 490, 48);
    text(s, "A qualified handoff requires", 620, 408, 430, 30, 21, C.ink, true);
    text(s, "• one finger released for at least 2 control steps\n• at least 2 other fingertips remain in contact\n• object stays inside the retention gate with no palm contact\n• released finger reconnects before timeout and speed recovers", 620, 458, 500, 150, 18, C.muted);
    runLabel(s, "V1  12-19-14 / 12-36-43     V2  14-58-36 / 15-47-22", 54, 626, 900, C.amber);
    notes(s, ["D:/UniLab/src/unilab/envs/manipulation/leap_inhand/finger_gaiting_rotation.py", "D:/UniLab/conf/ppo/task/leap_inhand_ball_finger_gaiting/mujoco.yaml", "D:/UniLab/DEVELOPMENT_LOG.md:2385"]);
  }

  // 13. Finger gaiting evidence
  {
    const s = deck.slides.add(); title(s, "Finger Gaiting Exposed Positive Rotation as the Bottleneck", 11);
    runLabel(s, "2026-07-22_15-47-22_mujoco", 54, 142);
    metric(s, 54, 192, 250, "22.33 s", "mean survival time", C.green);
    metric(s, 330, 192, 250, "10.48 mm", "mean anchor error", C.green);
    metric(s, 606, 192, 250, "2.12", "mean fingertip contacts", C.blue);
    metric(s, 882, 192, 250, "0.055%", "termination rate", C.blue);
    sectionBand(s, 380, "The gates that actually failed", C.red);
    text(s, "Positive speed gate", 78, 432, 300, 34, 22, C.ink, true);
    text(s, "1.97%", 78, 480, 220, 60, 38, C.red, true);
    text(s, "Qualified handoff", 430, 432, 300, 34, 22, C.ink, true);
    text(s, "< 0.11%", 430, 480, 220, 60, 38, C.red, true);
    text(s, "2s / 5s / 10s success", 782, 432, 320, 34, 22, C.ink, true);
    text(s, "all zero", 782, 480, 220, 60, 38, C.red, true);
    text(s, "Interpretation: forcing handoff before reliable positive rotation makes an already sparse exploration problem even harder.", 54, 610, 1130, 38, 22, C.ink, true);
    notes(s, ["D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallFingerGaitingRotation/2026-07-22_15-47-22_mujoco/run_config.json", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallFingerGaitingRotation/2026-07-22_15-47-22_mujoco/run_summary.json", "D:/UniLab/DEVELOPMENT_LOG.md:2624"]);
  }

  // 14. Direct reward
  {
    const s = deck.slides.add(); title(s, "Direct Rotation Isolates Basic +Z Acquisition", 12);
    text(s, "Fixed target: world +Z · 0.30 rad/s · cache reset · 25-second episode · no handoff objective", 54, 137, 1120, 28, 18, C.blue, true);
    rewardTable(s, [
      ["Positive term", "Weight"], ["stable_rotation", "+6.0"], ["rotation_streak", "+0.5"],
      ["thumb_contact", "+0.1"], ["fingertip_support", "+0.15"], ["center_recovery", "+1.0"],
    ], 54, 190, 500, 48);
    rewardTable(s, [
      ["Cost / penalty", "Weight"], ["stall", "-1.0"], ["reverse", "-4.0"], ["object_center", "-2.5"],
      ["object_linear_velocity", "-0.4"], ["palm_contact", "-1.0"], ["orthogonal_speed", "-0.25"],
      ["failure", "-5.0"],
    ], 620, 190, 540, 48);
    text(s, "Direct uses a different reward scale; aggregate return is comparable only among checkpoints from this task.", 54, 620, 1110, 36, 21, C.red, true);
    notes(s, ["D:/UniLab/conf/ppo/task/leap_inhand_ball_direct/mujoco.yaml", "D:/UniLab/src/unilab/envs/manipulation/leap_inhand/direct_rotation.py", "D:/UniLab/DEVELOPMENT_LOG.md:2620"]);
  }

  // 14. Direct evidence
  {
    const s = deck.slides.add(); title(s, "Direct Acquired Forward Spin, Not Sustained Motion", 13);
    runLabel(s, "2026-07-22_21-27-39_mujoco", 54, 142);
    text(s, "Rotation acquisition", 54, 200, 300, 30, 24, C.green, true);
    rewardTable(s, [
      ["Task metric", "Final value"], ["target +Z speed", "0.300 rad/s"],
      ["axis-speed EMA", "+0.246 rad/s"], ["axis purity", "84.3%"],
      ["mean anchor error", "14.7 mm"], ["mean fingertip contacts", "2.47"],
      ["palm contact rate", "0.0%"],
    ], 54, 246, 540, 46);
    text(s, "Sustainability check", 650, 200, 300, 30, 24, C.red, true);
    rewardTable(s, [
      ["Success gate", "Observed fraction"], ["continuous spin ≥ 2 s", "0.012%"],
      ["continuous spin ≥ 5 s", "0%"], ["continuous spin ≥ 10 s", "0%"],
      ["natural handoff rate", "0.061%"], ["final success fraction", "0.061%"],
    ], 650, 246, 500, 46);
    text(s, "Conclusion: the policy can generate positive +Z rotation, but it cannot maintain the motion, switch contacts, or complete multiple turns.", 54, 620, 1100, 42, 22, C.ink, true);
    notes(s, ["D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallDirectRotation/2026-07-22_21-27-39_mujoco/events.out.tfevents.1784780863.沅霆.41532.0", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallDirectRotation/2026-07-22_21-27-39_mujoco/run_config.json"]);
  }

  // 15. Cache gaiting
  {
    const s = deck.slides.add(); title(s, "Cache Gaiting Held Position but Did Not Spin or Handoff", 14);
    box(s, 54, 150, 500, 280, C.amberLight, "none", "roundRect");
    text(s, "V1", 84, 188, 90, 34, 26, C.amber, true);
    text(s, "Combined the simplified rotation objective with stable-support, release-progress, and handoff shaping.\n\nProblem: gaiting pressure increased exploration cost before stable rotation existed.", 84, 244, 410, 150, 18, C.ink);
    runLabel(s, "2026-07-27_12-09-05_mujoco", 84, 392, 390, C.amber);
    box(s, 626, 150, 520, 280, C.greenLight, "none", "roundRect");
    text(s, "V2", 656, 188, 90, 34, 26, C.green, true);
    text(s, "Kept the rotation objective, removed stable/release shaping, reduced handoff influence, and allowed release only for non-thumb fingers.\n\nGoal: acquire rotation first, then require contact switching.", 656, 244, 430, 150, 18, C.ink);
    runLabel(s, "2026-07-27_12-26-57_mujoco", 656, 392, 430, C.green);
    rewardTable(s, [
      ["Latest task metric", "Observed value"], ["position gate", "98.8%"],
      ["axis purity", "95.1%"], ["axis speed", "+0.001 rad/s"],
      ["net turns / episode", "0.0069"], ["speed > 0.10 rad/s", "0.73%"],
      ["qualified handoff rate", "0.006%"],
    ], 54, 470, 740, 30);
    text(s, "Interpretation", 860, 474, 220, 30, 24, C.red, true);
    text(s, "The object stayed near its anchor and the rotation axis was clean, but axis speed and qualified handoffs were almost zero.\n\nRetention is not sustained rotation.", 860, 524, 290, 112, 19, C.ink, true);
    notes(s, ["D:/UniLab/conf/ppo/task/leap_inhand_ball_cache_gaiting/mujoco.yaml", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallCacheGaitingRotation/2026-07-27_12-26-57_mujoco/events.out.tfevents.1785180421.沅霆.36252.0", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallCacheGaitingRotation/2026-07-27_12-26-57_mujoco/run_config.json"]);
  }

  // 17. Evidence matrix
  {
    const s = deck.slides.add(); title(s, "Each Task Answers a Different Diagnostic Question", 15);
    const cols = [54, 284, 545, 810, 1040];
    const widths = [230, 261, 265, 230, 170];
    const headers = ["Task", "Representative RunId", "Question", "Supported conclusion", "Evidence"];
    headers.forEach((h, i) => { box(s, cols[i], 150, widths[i], 48, C.ink); text(s, h, cols[i] + 10, 163, widths[i] - 20, 24, 16, C.white, true); });
    const rows = [
      ["Ball Cache", "07-21_17-29-49", "Does cache reset work?", "Integration is executable", "integration only"],
      ["Sustained Cache", "07-21_18-22-39", "Can many grasps rotate?", "Formal baseline established", "rotation metrics"],
      ["Retention Grey-zone", "07-21_22-58-39", "Why settle at 15–30 mm?", "Retention improved, spin did not", "A/B diagnosis"],
      ["Finger Gaiting", "07-22_15-47-22", "Is contact switching missing?", "Positive speed is the first bottleneck", "gate metrics"],
      ["Direct", "07-22_21-27-39", "Can forward spin be acquired?", "Acquisition is possible but brief", "duration metrics"],
      ["Sustained V3-A", "07-27_00-36-25", "Can reward be simplified?", "Direct task signals retained", "reward ablation"],
      ["Cache Gaiting V2", "07-27_12-26-57", "Can constrained handoff return?", "Retention works; rotation remains absent", "task metrics"],
    ];
    rows.forEach((row, r) => {
      const y = 198 + r * 58;
      row.forEach((cell, i) => {
        box(s, cols[i], y, widths[i], 58, r % 2 === 0 ? C.light : C.white, C.line);
        text(s, cell, cols[i] + 10, y + 12, widths[i] - 20, 36, 14.5, i === 4 && cell === "smoke" ? C.red : C.ink, i === 0 || i === 4);
      });
    });
    text(s, "Historical checkpoints must be interpreted with their saved run_config.json, not the current parent YAML.", 54, 625, 1120, 30, 19, C.blue, true);
    notes(s, ["D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallCacheRotation/*/run_config.json", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/*/run_config.json", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallFingerGaitingRotation/*/run_config.json", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallDirectRotation/*/run_config.json", "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallCacheGaitingRotation/*/run_config.json"]);
  }

  // 18. Conclusion
  {
    const s = deck.slides.add(); title(s, "Prove Sustained Rotation Before Reintroducing Handoff", 16);
    text(s, "Three defensible conclusions", 54, 150, 440, 32, 24, C.ink, true);
    text(s, "1  A ball-specific reset cache and physically valid collision model are prerequisites.\n\n2  Better retention does not imply sustained rotation; signed axis speed and net angle must be measured.\n\n3  Handoff is eventually necessary, but forcing it before rotation acquisition increases exploration sparsity.", 54, 208, 560, 275, 20, C.ink);
    box(s, 690, 145, 455, 380, C.blueLight, "none", "roundRect");
    text(s, "Validation order", 720, 176, 360, 34, 26, C.blue, true);
    text(s, "01  Reward ablation with fixed cache and seed\n02  Report axis-speed duration and net turns\n03  Use the viewer to reject palm support or penetration\n04  Add handoff only after acquisition is stable\n05  Compare success rate across at least three seeds\n06  Add domain randomization last", 720, 244, 370, 220, 18, C.ink);
    text(s, "Success criteria", 54, 520, 220, 28, 22, C.green, true);
    text(s, "Sustained positive rotation · multiple net turns · low orthogonal speed · no drop · qualified handoff · cross-seed stability", 54, 566, 1090, 58, 22, C.ink, true);
    notes(s, ["D:/UniLab/AGENTS.md#Dexterous-Manipulation-Training-Guidance", "D:/UniLab/DEVELOPMENT_LOG.md"], "Maintain a conservative conclusion: the bottleneck is localized, but sustained Cache Gaiting is not yet solved.");
  }

  const sourceNotes = `LEAP Hand Monthly Progress Report\nGenerated from local UniLab evidence.\n\nPrimary sources:\n- D:/UniLab/AGENTS.md\n- D:/UniLab/DEVELOPMENT_LOG.md\n- D:/UniLab/conf/ppo/task/leap_inhand_ball_*/mujoco.yaml\n- D:/UniLab/src/unilab/envs/manipulation/leap_inhand/*.py\n- D:/UniLab/logs/rsl_rl_ppo/*/run_config.json\n- D:/UniLab/logs/rsl_rl_ppo/*/run_summary.json\n`;
  await fs.writeFile(`${TMP}/source-notes.txt`, sourceNotes, "utf8");

  for (const [index, slide] of deck.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await deck.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(`${TMP}/rendered/${stem}.png`, new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${TMP}/rendered/${stem}.layout.json`, await layout.text(), "utf8");
  }

  const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(`${TMP}/rendered/montage.webp`, new Uint8Array(await montage.arrayBuffer()));
  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(OUT);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
