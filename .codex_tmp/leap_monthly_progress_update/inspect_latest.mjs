import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const source = "D:/UniLab/LEAP_Hand_Monthly_Progress_Report_English_14pt_CacheFlow_Updated.pptx";
const outDir = "D:/UniLab/.codex_tmp/leap_monthly_progress_update/before";

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

const deck = await PresentationFile.importPptx(await FileBlob.load(source));
await fs.mkdir(outDir, { recursive: true });

const inspection = await deck.inspect({
  kind: "slide,textbox,shape,image,table,chart,notes,layout",
  include: "id,slide,name,title,text,textPreview,bbox,bboxUnit,isPlaceholder,placeholders",
  maxChars: 50000,
});
await fs.writeFile(`${outDir}/inspection.ndjson`, inspection.ndjson, "utf8");

for (let index = 0; index < deck.slides.items.length; index += 1) {
  const slide = deck.slides.getItem(index);
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  await writeBlob(`${outDir}/${stem}.png`, await deck.export({ slide, format: "png", scale: 1 }));
  await fs.writeFile(`${outDir}/${stem}.layout.json`, await (await slide.export({ format: "layout" })).text());
}

const help = deck.help("*", {
  search: "slide.shapes.delete|shape.delete|slide.shapes.add|deleteAll|remove shape",
  include: ["index", "examples", "notes"],
  maxChars: 12000,
});
await fs.writeFile(`${outDir}/delete-help.ndjson`, help.ndjson, "utf8");
console.log(`slides=${deck.slides.items.length}`);
console.log(inspection.ndjson.split("\n").filter((line) => /\"slide\":(7|8),/.test(line)).join("\n"));
console.log(help.ndjson);
