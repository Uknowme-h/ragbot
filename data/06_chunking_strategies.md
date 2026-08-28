# Document Chunking Strategies

Chunking decides what a single embedding represents. It is the most
common reason a RAG demo "doesn't find" an answer that is clearly in the
PDF.

## Fixed-size windows with overlap

`RecursiveCharacterTextSplitter` walks a separator list (`\n\n`, `\n`,
`. `, `" "`, `""`) and cuts near `chunk_size` characters. Overlap repeats the
boundary so a sentence that straddles two chunks is still complete in at
least one of them.

This service uses **chunk_size=800** and **chunk_overlap=150** characters.
For technical Markdown that is roughly a short section: large enough for a
code sample plus explanation, small enough that similarity is not washed
out by unrelated paragraphs.

## Why overlap matters

Without overlap, a definition at the end of chunk 4 and its example at
the start of chunk 5 may never appear together in the prompt. 10–20%
overlap is the usual range. More overlap means more stored vectors and
more duplicate text in top-k.

## Structure-aware splits

Prefer splitting on headings before falling back to character counts.
Markdown documents in this knowledge base start with `#` / `##` headings;
the recursive splitter already favors blank lines, which usually align
with those headings.

For PDFs, page boundaries are stored as metadata even when a chunk does
not align with a page. A chunk that begins on page 3 and spills into page
4 should keep `page_number=3` (the start page) so citations stay stable.

## Tables and code

Tables extracted from PDFs become ragged plain text. That is a known
limitation: cell alignment is lost. Put important numbers in surrounding
prose if you need them retrieved reliably.

Code fences should stay intact. A splitter that cuts through the middle of
a function makes the embedding nearly useless. The 800-character budget
usually fits one example function from these docs.

## Re-chunking

If evaluation queries miss, inspect the retrieved chunk text before
changing the embedding model. Typical fixes:

- Increase chunk size when answers need two adjacent paragraphs.
- Decrease chunk size when a 2000-character chunk mixes two topics.
- Add overlap if definitions are truncated.

Changing chunk size requires a full re-ingest. The ingest CLI supports
`--reset` to drop the collection first.
