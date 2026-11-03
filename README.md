# Grid Inspection Ops

Production-oriented quality-inspection workflow for the MVTec `grid` case.

Project progress is tracked on the [V1 GitHub Project board](https://github.com/users/cankiziloz/projects/1/views/1?system_template=kanban).

The [portfolio operational documents](docs/README.md) define the intended
camera-to-API integration and human-review workflow. The file-upload control is
only a local demo/test harness; a production deployment would receive images
from an automated camera or edge gateway.

## Retrieval-augmented procedures

The default local mode uses deterministic lexical retrieval over the approved
Markdown SOPs, so the project works without cloud credentials. Production-like
mode uses Pinecone's integrated embeddings and current Documents API: each
reviewed SOP section is stored as a source-text document; Pinecone embeds that
text on ingestion and embeds the query on semantic search. The policy reason
code remains a metadata filter, so
semantic similarity can select a section *only inside the SOP documents that
the deterministic policy has approved*. Retrieval informs the explanation; it
can never change a visual decision.

Provision a Pinecone integrated-embedding index with embedding field `chunk_text` (for
example using `llama-text-embed-v2`), then use its host and a namespace:

```bash
pip install -e '.[dev,rag]'
export PINECONE_API_KEY=...          # keep this out of Git
export PINECONE_INDEX_HOST=...       # copied from the Pinecone index details
export PINECONE_NAMESPACE=grid-inspection-sop-v1
export PINECONE_EMBEDDING_FIELD=chunk_text
python scripts/sync_pinecone_sops.py # controlled SOP deployment step
export RETRIEVAL_PROVIDER=pinecone
```

`sync_pinecone_sops.py` is idempotent for unchanged sections: stable IDs cause
an approved section to be upserted rather than duplicated. After any reviewed
SOP change, run it in CI/CD or as a deliberate release operation before
deploying the API. Do not set `RETRIEVAL_PROVIDER=pinecone` until the sync has
succeeded. The inspection runtime fails closed if its credentials/host are absent.
For local development these three values can instead live in the ignored `.env`
file using `KEY=value` syntax. The application loads this file automatically;
already-exported environment variables take precedence.

Run the local keyword/natural-language retrieval regression benchmark with:

```bash
python scripts/evaluate_retrieval.py
```

## Optional local Ollama explanations

The default is the deterministic template explainer. To enable a local Ollama
generator after installing a model, configure:

```bash
export EXPLANATION_PROVIDER=ollama
export OLLAMA_MODEL=qwen2.5:1.5b
uvicorn app.api:app --reload
```

The generator receives only the immutable decision, measured evidence, and
retrieved SOP excerpts. Its JSON output is validated; unapproved citations,
malformed output, or an unavailable Ollama service fall back to the template.
Check the real local integration with `python scripts/smoke_ollama.py`.

The CV model is the only component allowed to choose `PASS`, `INSPECT`, or
`REJECT`. LangGraph orchestrates validation, durable human review, audit events,
history checks, and LLM-generated explanations/checklists. An LLM never changes
the visual decision.

## Architecture

```text
upload -> validate -> Grid STN + PatchCore -> deterministic policy
                                             -> PASS    -> audit + summary
                                             -> INSPECT -> human review
                                             -> REJECT  -> audit + alerting
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
export GRID_MODEL_REPO_ID=YOUR_HF_ACCOUNT/grid-inspection-patchcore-v1
# Keep this at main while iterating; pin it to a validated commit SHA before deployment.
export GRID_MODEL_REVISION=main
uvicorn app.api:app --reload
pytest
```

On its first inspection request, the API downloads the published Hugging Face
release, validates every artifact checksum in `manifest.json`, then loads it.
Model files are cached locally and are never committed to this code repository.
A production deployment should use PostgreSQL, object storage for images/masks,
and a worker queue for inference; SQLite is intended only for local development.

To validate a download without running the API:

```bash
python scripts/download_model.py --repo-id "$GRID_MODEL_REPO_ID"
```

## Train and create a versioned model release

The training code is deliberately separate from the API. It uses only
`grid/train/good` to build PatchCore memory banks, holds out a deterministic
normal validation split to calibrate the two decision thresholds, and uses the
test split only for the exported evaluation report.

```bash
python -m app.training.train \
  --dataset-root /absolute/path/to/data \
  --output releases/grid-patchcore-v1 \
  --seed 42 --review-percentile 95 --reject-percentile 99.5
```

The resulting folder contains `grid_stn.pth`, `patchcore_model/`,
`evaluation.csv`, and a checksummed `manifest.json`. It is ignored by Git on
purpose: it is a model release, not source code.

To publish that exact release, first copy `model-card-template.md` into the
release folder as `README.md`, complete its evaluation section, then run:

```bash
pip install -e '.[publish]'
HF_TOKEN=... python scripts/publish_model.py \
  --release releases/grid-patchcore-v1 \
  --repo-id YOUR_HF_ACCOUNT/grid-inspection-patchcore-v1
```

MVTec AD is CC BY-NC-SA 4.0. Keep its attribution and non-commercial terms in
the model card; do not present MVTec results as production validation.

### Package the already validated model (recommended first release)

Do not retrain a model whose behaviour you want to preserve. This command only
copies the existing checkpoint, memory banks, thresholds, and evaluation CSV;
it does not modify or recalibrate any source artifact.

```bash
python scripts/package_existing_model.py \
  --stn /Users/cankiziloz/Documents/industrial-vision-inspection/mvtec_grid_stn.pth \
  --patchcore /Users/cankiziloz/Documents/industrial-vision-inspection/patchcore_model \
  --evaluation /Users/cankiziloz/Documents/industrial-vision-inspection/model/results/grid_case_accuracy.csv \
  --output releases/grid-patchcore-v1
```
