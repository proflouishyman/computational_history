# Probate Audit Logic (Overview)

This pipeline audits historical probate ledger pages by reconstructing per-page totals and verifying running totals across pages. It prioritizes accurate entry summation, avoids double-counting, and limits audits to ledger-like pages (not narrative inventory sections).

## 1. Ingestion and Page Parsing
- Reads `combined.txt`, splitting into per-page JSON objects.
- Builds a `pages_df` table with:
  - OCR text (`corrected_ocr`)
  - page number
  - metadata fields
  - a derived `page_type` (`ledger`, `inventory`, or `unknown`)
  - a flag for section headers (e.g., `Book Debts`, `Inventory`, `Description No.`)

## 2. Amount Extraction
For each line in each page:
- Extracts numeric amounts using a flexible regex:
  - supports `$`/`£`, commas, decimals, and fractions (e.g., `77 1/3`)
- Filters out non-amount numbers:
  - years (e.g., 1839)
  - page references (`page 94`, `p. 94`)
  - date strings (month + `day` patterns)

Each amount is labeled with a `role`:
- `total_bf` — amount brought forward
- `total_cf` — amount carried forward
- `section_total` — `Total amount of ...` lines (excluded from item sums)
- `item` — genuine ledger line item

## 3. Page Totals (BF/CF)
- Detects BF and CF totals by scanning text near `Amount Brought Forward` / `Amount Carried Forward`.
- Records whether multiple BF/CFs appear or totals are incomplete.

## 4. Item Summation
- Only `role == item` contributes to `item_sum`.
- Section totals, BF/CF totals, and date/page numbers are excluded.
- Continuation logic attempts to avoid double counting repeated entries across page breaks.

## 5. Page Type Filtering (Audit Eligibility)
Pages are eligible for audit only if:
- `page_type == ledger`
- no section header detected (to avoid subsections like `Book Debts`)
- BF and CF are both present

Inventory/real-estate narrative pages are excluded from arithmetic checks.

## 6. Expected CF and Delta
For audit-eligible pages:
```
expected_cf = bf + item_sum
delta = cf - expected_cf
```
`delta` drives anomaly detection.

## 7. Anomaly Classification
If `abs(delta) > tolerance`:
- `missing_or_doublecount_item` if delta matches a single item
- `continuation_or_dedup` if a continuation indicator exists
- `fraction_drift` for small fractional errors
- `ocr_separator_artifact` for OCR thousand/decimal corruption
- `unexplained` otherwise

Non-ledger and section-boundary pages are explicitly labeled so they do not pollute the audit.

## 8. Outputs
Outputs are written to `audit_output`:
- `ledger_audit.csv.gz` (main audit table)
- `line_items.csv.gz`
- `page_totals.csv.gz`
- `audit_report.md` (human-readable summary)
- Parquet equivalents if a Parquet engine is installed
