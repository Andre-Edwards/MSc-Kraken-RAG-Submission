# Corpus notes

The case-study corpus contains public official Kraken material only. It does not contain internal policies, customer records or confidential company information.

## PDF corpus

The `kraken_PDFs` folder contains 12 documents used during development. `pdf_corpus_manifest.csv` records each filename, file size and SHA-256 hash so the submitted files can be checked.

One vendor conflict document is image-based. The current PyMuPDF ingestion pipeline does not perform OCR, so that document may not produce usable chunks. I treated this as a corpus coverage limitation rather than a fair retrieval failure.

## Web corpus

The local development crawl contained 64 saved records representing 63 unique URLs. The raw page text is not committed to this clean repository because public pages can change and copying the whole snapshot is not necessary to run the application.

Instead, `web_sources/web_seed_urls.txt` lists the unique URLs and `web_sources/web_corpus_manifest.csv` records the title, crawl timestamp when available, text length and SHA-256 hash of the extracted text used locally.

Recrawling these URLs may produce a different corpus if Kraken changes or removes a page. The saved evaluation results document the corpus state used for the reported experiment.

The web crawler checks `robots.txt`, limits itself to configured domains and applies a delay between requests. A large or unrestricted crawl is not required to reproduce the central chunking experiment.
