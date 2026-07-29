import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "C:/Users/admin/Documents/华为算子比赛/.tmp_s9_official_20260722/算子挑战赛S9赛题/S9挑战性能赛题.xlsx";
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const sheets = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 8000,
});
console.log("SHEETS");
console.log(sheets.ndjson);

const overview = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 30000,
  tableMaxRows: 40,
  tableMaxCols: 20,
  tableMaxCellChars: 300,
});
console.log("OVERVIEW");
console.log(overview.ndjson);
