# Data/

Small **reference data** lives here and is committed to the repository:

* `milletvekilleri.csv`, `milletvekilleri.json` — canonical MP roster (name ↔ party ↔ term),
  used for speaker/party resolution.
* `data.txt` — auto-generated summary of the raw corpus (shape, unique counts, missingness).

The two **large raw corpus files** (~919 MB each) are **not** stored on GitHub because they
exceed the 100 MB/file limit:

* `TBMM_Network_Dataset.csv`
* `TBMM_Network_Dataset_partyfixed.csv`

Download them from Google Drive and place them in this folder:

> https://drive.google.com/drive/folders/1Z1zCytImXABvIRBK7msO6vorx7DfUn2t

They are required only for **full reproduction from raw transcripts**. The published figures
regenerate from the committed processed CSVs (`Term_*/CSVs/`, `Inter_Term/CSVs/`) without them.
See [`../docs/DATA.md`](../docs/DATA.md) for full schema and provenance.
