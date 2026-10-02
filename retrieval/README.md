# Retrieval data

From the repository root, start Milvus 2.5.6 with the included script:

```bash
bash retrieval/milvus.sh start
```

The script resolves paths relative to itself, uses `retrieval/volumes/milvus`,
and binds ports to localhost. It preserves existing YAML configuration and
waits up to 180 seconds for health. Set `MILVUS_URI=http://localhost:19530` and
`DEVICE=cpu` (or `cuda`) in `.env`.

Use `bash retrieval/milvus.sh stop`, `status`, or `remove`; removal only deletes a
stopped container and retains its data. `MILVUS_CONTAINER` and `MILVUS_DATA_DIR`
select another container name or absolute data directory. To change mounts,
stop/remove the old container, then start it with the new settings.

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
python retrieval/upload_corpus.py
# Or selected sources:
python retrieval/upload_corpus.py --corpus cpg textbooks
```

This inserts data into `cpg`, `recop`, `textbooks`, and `statpearls`. Use a fresh
Milvus database when rebuilding; rerunning insertion against restored data can
duplicate rows. Retrieval uses the MedCPT query encoder and cross encoder.
