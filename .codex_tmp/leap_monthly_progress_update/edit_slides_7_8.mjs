import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const SOURCE = "D:/UniLab/LEAP_Hand_Monthly_Progress_Report_English_14pt_CacheFlow_Updated.pptx";
const OUT = "D:/UniLab/LEAP_Hand_Monthly_Progress_Report_English_14pt_CacheFlow_Performance_Updated.pptx";
const TMP = "D:/UniLab/.codex_tmp/leap_monthly_progress_update";
const W = 1280;
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

function text(slide, value, x, y, w, h, size = 19, color = C.ink, bold = false, align = "left", autoFit = "shrinkText") {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    fontSize: Math.max(size, 19),
    typeface: FONT,
    color,
    bold,
    alignment: align,
    verticalAlignment: "middle",
    autoFit,
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
  slide.speakerNotes.textFrame.setText(
    `${presenter}\n\n[Sources]\n${sources.map((source) => `- ${source}`).join("\n")}`,
  );
  slide.speakerNotes.setVisible(true);
}

function table(slide, { x, y, widths, headers, rows, rowHeight, tones }) {
  const headerHeight = 36;
  let xx = x;
  headers.forEach((header, i) => {
    box(slide, xx, y, widths[i], headerHeight, C.ink, C.white);
    text(slide, header, xx + 6, y, widths[i] - 12, headerHeight, 19, C.white, true, i >= 3 ? "center" : "left", "none");
    xx += widths[i];
  });

  rows.forEach((row, rowIndex) => {
    const yy = y + headerHeight + rowIndex * rowHeight;
    let cellX = x;
    row.forEach((value, columnIndex) => {
      const fill = rowIndex % 2 === 0 ? C.light : C.white;
      box(slide, cellX, yy, widths[columnIndex], rowHeight, fill, C.line);
      if (columnIndex === 0) {
        box(slide, cellX + 6, yy + 7, widths[columnIndex] - 12, rowHeight - 14, tones[rowIndex], "none", "roundRect");
        text(slide, value, cellX + 6, yy + 7, widths[columnIndex] - 12, rowHeight - 14, 19, C.white, true, "center", "none");
      } else {
        const numeric = columnIndex >= 3;
        text(slide, value, cellX + 6, yy + 3, widths[columnIndex] - 12, rowHeight - 6, 19, columnIndex === 1 ? C.blue : C.ink, columnIndex === 1, numeric ? "center" : "left");
      }
      cellX += widths[columnIndex];
    });
  });
}

function callout(slide, x, y, w, h, heading, body, fill, tone) {
  box(slide, x, y, w, h, fill, "none", "roundRect");
  text(slide, heading, x + 18, y + 12, w - 36, 28, 19, tone, true, "left", "none");
  text(slide, body, x + 18, y + 44, w - 36, h - 54, 19, C.ink, false, "left", "shrinkText");
}

function buildSlide7(slide) {
  slide.shapes.deleteAll();
  slide.background.fill = C.white;
  title(slide, "S1–S6: Anchor Helped, but Rotation Still Stalled", 7);
  text(slide, "Last 10% iteration mean · axis speed is EMA", 54, 111, 700, 24, 19, C.muted, false, "left", "none");

  const rows = [
    ["S1", "07-21 18-22-39", "Baseline curriculum", "21.83", "2.21", "16.08%", ".0128", ".59%", "12.06"],
    ["S2", "07-21 22-58-39", "+ anchor proximity", "19.87", "3.05", "17.69%", ".0137", "2.11%", "12.14"],
    ["S3", "07-22 02-38-07", "+ level spin floor", "15.76", "3.36", "10.56%", ".0174", ".15%", "7.87"],
    ["S4", "07-22 23-04-12", "Direct spin; anchor off", "18.04", "2.23", "13.01%", ".0161", "0%", "10.81"],
    ["S5", "07-23 00-19-50", "Fixed 0.30 rad/s", "14.49", "2.57", ".07%*", ".0255", "0%", "9.88"],
    ["S6", "07-23 02-40-23", "+ continuity penalty", "10.06", "2.71", ".23%*", ".0103", "0%", "8.51"],
  ];
  table(slide, {
    x: 54,
    y: 142,
    widths: [70, 216, 264, 82, 70, 94, 92, 78, 100],
    headers: ["Stage", "Effective Run", "Reward change", "Error\n(mm)", "Tips", "Stage\nvalid", "Axis\nEMA", "≥2 s", "Survival\n(s)"],
    rows,
    rowHeight: 47,
    tones: [C.blue, C.green, C.cyan, C.amber, C.red, C.blue],
  });

  callout(slide, 54, 476, 352, 158, "Anchor was the clean win", "S1→S2: error −9%, fingertip contacts +38%, and ≥2 s success rose 3.6× with unchanged survival.", C.greenLight, C.green);
  callout(slide, 423, 476, 352, 158, "Closer did not mean sustained", "S3 and S6 pulled the ball closer, but ≥2 s success collapsed and axis speed remained far below the 0.30 rad/s target.", C.amberLight, C.amber);
  callout(slide, 792, 476, 354, 158, "No long-duration solution", "No S1–S6 run achieved sustained ≥5 s. *S5–S6 changed stage semantics, so stage-valid is not directly comparable.", C.redLight, C.red);

  notes(slide, [
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-21_18-22-39_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-21_22-58-39_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-22_02-38-07_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-22_23-04-12_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-23_00-19-50_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-23_02-40-23_mujoco",
  ], "Metrics are arithmetic means over the final 10% of TensorBoard iteration records. S5–S6 use fixed-speed stage semantics and are flagged where direct comparison is unsafe.");
}

function buildSlide8(slide) {
  slide.shapes.deleteAll();
  slide.background.fill = C.white;
  title(slide, "S7–S13: Fast Bursts Gave Way to Stable Holding", 8);
  text(slide, "Last 10% iteration mean · axis speed is raw, not EMA · sustained-duration metrics were not logged", 54, 111, 1100, 24, 19, C.muted, false, "left", "none");

  const rows = [
    ["S7", "07-23 03-57-44", "Minimal Allegro-style rotate", "—", "—", ".361", "6.79"],
    ["S8", "07-23 15-34-09", "+ object linear velocity", "—", "—", ".394", "5.53"],
    ["S9", "07-23 20-18-51", "+ position error", "15.52", "1.59", ".484", "5.62"],
    ["S10", "07-23 21-54-51", "+ contact-weighted rotation", "17.03", "2.86", ".384", "3.24"],
    ["S11", "07-25 03-50-30", "+ support pose / opposition", "22.68", "1.39", ".471", "7.00"],
    ["S12", "07-26 23-27-56", "V3-A: retention-first gate", "6.70", "2.75", ".030", "24.56"],
    ["S13", "07-27 00-36-25", "+ gated stall penalty", "6.50", "1.97", ".057", "24.67"],
  ];
  table(slide, {
    x: 54,
    y: 142,
    widths: [70, 216, 468, 90, 80, 112, 100],
    headers: ["Stage", "Effective Run", "Reward change", "Error\n(mm)", "Tips", "Raw axis\n(rad/s)", "Survival\n(s)"],
    rows,
    rowHeight: 42,
    tones: [C.blue, C.cyan, C.green, C.amber, C.red, C.blue, C.green],
  });

  callout(slide, 54, 490, 352, 144, "Contact gating backfired", "S9→S10 increased contacts 1.59→2.86, but reduced speed and survival. More contact was not better contact sequencing.", C.amberLight, C.amber);
  callout(slide, 423, 490, 352, 144, "V3-A solved retention", "S11→S12 cut error 22.68→6.70 mm and raised survival 7.00→24.56 s, while raw spin collapsed 0.471→0.030.", C.blueLight, C.blue);
  callout(slide, 792, 490, 354, 144, "Stall pressure recovered speed", "S12→S13 nearly doubled raw axis speed without harming retention or survival, but duration evidence was still missing.", C.greenLight, C.green);

  notes(slide, [
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-23_03-57-44_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-23_15-34-09_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-23_20-18-51_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-23_21-54-51_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-25_03-50-30_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-26_23-27-56_mujoco",
    "D:/UniLab/logs/rsl_rl_ppo/LeapInhandBallSustainedCacheRotation/2026-07-27_00-36-25_mujoco",
  ], "Metrics are arithmetic means over the final 10% of TensorBoard iteration records. These runs log raw axis speed, not axis-speed EMA, and do not expose sustained-duration fractions; high raw speed plus short survival is interpreted as burst rotation rather than sustained rotation.");
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(`${TMP}/after`, { recursive: true });
  const deck = await PresentationFile.importPptx(await FileBlob.load(SOURCE));
  if (deck.slides.items.length !== 16) {
    throw new Error(`Expected 16 slides, found ${deck.slides.items.length}`);
  }

  buildSlide7(deck.slides.items[6]);
  buildSlide8(deck.slides.items[7]);

  for (const [index, slide] of deck.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(`${TMP}/after/${stem}.png`, await deck.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${TMP}/after/${stem}.layout.json`, await layout.text());
  }

  await writeBlob(`${TMP}/after/deck-montage.webp`, await deck.export({ format: "webp", montage: true, scale: 1 }));
  const inspection = await deck.inspect({ kind: "slide,textbox,shape,notes,layout", maxChars: 60000 });
  await fs.writeFile(`${TMP}/after/inspection.ndjson`, inspection.ndjson);

  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(OUT);
  console.log(`Updated ${OUT}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
