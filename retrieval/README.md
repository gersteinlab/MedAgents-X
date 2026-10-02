# Retrieval data

From the repository root, start Milvus 2.5.6 with the included script:

```bash
cd retrieval
bash standalone_embed.sh start
```

The script uses `retrieval/volumes/milvus`. Always run it from `retrieval/`.
Set `MILVUS_URI=http://localhost:19530` and `DEVICE=cpu` (or `cuda`) in `.env`.
Stop it with `bash standalone_embed.sh stop`.

## Restore an existing database

Stop Milvus before restoring its data. Extract a cold backup into an empty
working directory and mount its `volumes/milvus` directory with the same Milvus
version and embedded etcd configuration. Keep the original archive and verify
its SHA-256 first. Database directories under the project root and under
`retrieval/` are separate snapshots; never merge them.

## Rebuild collections from corpus

Each corpus lives under `retrieval/corpus/<name>/`:

- `data/`: source chunks
- `vector/`: precomputed 768-dimensional embeddings
- `json/`: text and metadata associated with embeddings

```bash
cd retrieval
bash upload_all.sh
```

This inserts data into `cpg`, `recop`, `textbooks`, and `statpearls`. Use a fresh
Milvus database when rebuilding; rerunning insertion against restored data can
duplicate rows. Retrieval uses the MedCPT query encoder and cross encoder.
