import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const INPUT = "D:/UniLab/LEAP_Hand_Monthly_Progress_Report_English_14pt_CacheFlow_FingerGaiting_Updated.pptx";
const OUT = "D:/UniLab/.codex_tmp/leap_monthly_progress_update/final_check_finger_gaiting";
const W = 1280;
const H = 720;

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(OUT, { recursive: true });
  const deck = await PresentationFile.importPptx(await FileBlob.load(INPUT));
  const failures = [];

  for (const [index, slide] of deck.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(`${OUT}/${stem}.png`, await deck.export({ slide, format: "png", scale: 1 }));
    const layoutBlob = await slide.export({ format: "layout" });
    const layoutText = await layoutBlob.text();
    await fs.writeFile(`${OUT}/${stem}.layout.json`, layoutText);
    const layout = JSON.parse(layoutText);
    for (const element of layout.elements ?? []) {
      const [x, y, w, h] = element.bbox ?? [];
      if (![x, y, w, h].every(Number.isFinite)) continue;
      if (x < -0.5 || y < -0.5 || x + w > W + 0.5 || y + h > H + 0.5) {
        failures.push({ slide: index + 1, id: element.aid, bbox: element.bbox, text: element.textPreview ?? "" });
      }
    }
  }

  const inspection = await deck.inspect({ kind: "slide,textbox,shape,notes,layout", maxChars: 100000 });
  await fs.writeFile(`${OUT}/inspection.ndjson`, inspection.ndjson);
  await fs.writeFile(`${OUT}/validation.json`, JSON.stringify({ slideCount: deck.slides.items.length, canvas: [W, H], overflowFailures: failures }, null, 2));

  if (deck.slides.items.length !== 16 || failures.length > 0) {
    throw new Error(`Validation failed: slides=${deck.slides.items.length}, overflow=${failures.length}`);
  }
  console.log("Validation passed: 16 slides, no out-of-canvas elements.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
