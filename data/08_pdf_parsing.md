# PDF Parsing Pitfalls

PDFs are a print format, not a document format. A robust ingest pipeline
assumes every file might be broken in a different way.

## Text layer vs scanned pages

A digital PDF has a text layer. `page.get_text("blocks")` returns tuples
`(x0, y0, x1, y1, text, block_no, block_type)`. Sort by `y` then `x` to
approximate reading order for multi-column layouts (left column then
right column if you sort purely by y; for true two-column, bucket by x
midpoint first).

A scanned PDF has **no text layer**. `get_text()` returns an empty string.
The fallback is OCR: rasterize the page (`page.get_pixmap(dpi=200)`),
run Tesseract (`pytesseract.image_to_string`), and continue. If Tesseract
is not installed, log a warning and skip the page rather than crashing
ingest.

## Password-protected and corrupt files

`fitz.open()` raises `fitz.FileDataError` (or a `RuntimeError`) for
encrypted or truncated files. Catch it, log the path, and skip. Never
abort the whole ingest run because one file is unreadable.

## Memory

Do not concatenate every page of a 400-page PDF into one string. Iterate
pages, clean, chunk, and accumulate chunks. The embedding step already
batches at 32; page-level generators keep RAM flat.

## Layout artifacts to clean

- Standalone page numbers (`^\d+$` on their own line)
- Null bytes (`\x00`) that some extractors emit
- Repeated running headers ("Confidential — Page N")
- Ligatures and hyphenation at line breaks (`embed-\ning` → `embedding`)

This pipeline strips page-number-only lines, null bytes, and collapsed
runs of whitespace. Hyphenation repair is best-effort and conservative
(only when a line ends with `-` and the next line starts lowercase).

## Tables

Cell grids become linear text. Retrieval can still find a number if it
appears as a contiguous token, but "the value in row 3, column 2" queries
will fail. Document this limitation; do not pretend PDF tables are
structured data without a table parser (e.g. Camelot) which is out of
scope for this assessment.

## Markdown and mixed corpora

The same chunker must accept `.md` and `.txt` by reading UTF-8 and treating
the file as a single page (`page_number=1`). Mixing PDFs and Markdown in
one collection is fine as long as metadata always has `source_file`.

## Idempotent ingest

Hash file bytes. If `data/07_groq_inference.md` is unchanged, skip it. If
the file changed, delete existing chunks whose metadata `file_hash`
matches the old hash for that `source_file`, then insert the new ones.
The CLI flag `--reset` drops the entire collection when you want a clean
rebuild.
