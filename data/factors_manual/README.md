# Manual factor series (drop-in)

This folder is the ingestion path for series with no free API — ISM prints,
semi book-to-bill, DRAM spot prices, compute futures (until they get a live
feed), anything hand-collected from a Bloomberg terminal or a PDF.

## How to add a series

1. Save a CSV here with exactly two columns:

   ```csv
   date,value
   2025-09-01,47.2
   2025-10-01,48.1
   ```

2. Register it in `macro_loader.MACRO_FACTORS` with `"source": "manual"`,
   the `"file"` name, a `"transform"` (`level_change` for indices/rates,
   `log_return` for prices), and `"frequency": "monthly"` if it prints
   monthly (monthly series are held across the month and excluded from
   factor PCA, which needs daily variance).

3. Re-run the pipeline. Manual files are re-read every run (no cache), so
   updating a CSV takes effect immediately.

`ISM_PMI.csv` is pre-registered — drop the file in and the actual ISM series
appears next to its market proxy (CYCDEF) on the next run.

CSVs in this folder ARE committed to git on the next daily refresh — that is
deliberate, so hand-collected data survives and ships to the dashboard.
