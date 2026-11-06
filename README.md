# Grid Inspection Ops

Production-oriented quality-inspection workflow for the MVTec `grid` case.

Project progress is tracked on the [V1 GitHub Project board](https://github.com/users/cankiziloz/projects/1/views/1?system_template=kanban).

The [portfolio operational documents](docs/README.md) define the intended
camera-to-API integration and human-review workflow. The file-upload control is
only a local demo/test harness; a production deployment would receive images
from an automated camera or edge gateway.

## Production-like demo feed

The dashboard is designed around automated ingress. Use the simulator to send
a shuffled MVTec Grid test corpus at a fixed cadence; it supplies the same
station/camera/frame/checksum metadata that an edge gateway would send. The
folder name is printed by the simulator only as demo ground truth and is never
sent to the model.

```bash
python scripts/simulate_edge_feed.py \
  --dataset-root /absolute/path/to/data/grid/test \
  --interval-seconds 5
```

Use `--interval-seconds 10` for a slower feed, and `--max-events 20` for a
finite demo. Each frame has a unique `frame_id`; the `/v1/edge/inspections`
endpoint validates the image SHA-256 and returns the original result on a
duplicate frame instead of creating another inspection.

Run the API and dashboard in separate terminals, then start the simulator in a
third terminal:

```bash
uvicorn app.api:app --reload
python -m http.server 8080 --directory frontend
python scripts/simulate_edge_feed.py --dataset-root /absolute/path/to/data/grid/test
```

Open the dashboard at `http://localhost:8080` (or `http://127.0.0.1:8080`).
The API accepts both local origins by default; override them for a deployed UI
with the comma-separated `CORS_ORIGINS` environment variable.

The dashboard refreshes every 10 seconds and shows 24-hour, 7-day, 30-day, or
all-time inspection activity; review outcomes; open review backlog; policy
reason codes; and latency p50/p95 measurements.

## V2 demo feedback seed

V2 starts with an auditable case-memory corpus, not an automatic decision
override. To mark only pending **demo** glue INSPECT records as human-accepted
process residue, run:

```bash
python scripts/seed_v2_glue_residue_feedback.py
```

The script records `PASS + ACCEPTED_PROCESS_RESIDUE`, adds a `glue` case-memory
annotation, and writes an audit event. It never changes a non-demo record, an
automatic PASS/REJECT result, or an existing human resolution; rerunning it is
safe.

## Operations briefing snapshot

The optional Operations Briefing is built in stages. Its first stage is a
deterministic, typed snapshot—not an LLM response. It packages the selected
dashboard period's activity, review outcomes, queue, policy reasons, review
dispositions, latency metrics, and deterministic SOP evidence for a later
constrained generator. Each evidence item is an approved SOP section selected
by the policy-reason routing table, with its source reason code and related
event count; it is not an unrestricted document search.

```bash
curl "http://127.0.0.1:8000/v1/operations/briefing-snapshot?period=24h"
```

Supported periods are `24h`, `7d`, `30d`, and `all`. The snapshot endpoint
does not call Ollama and cannot change any inspection decision.

## Optional Operations Briefing narrator

`/v1/operations/briefing` turns the immutable snapshot into a bounded shift
summary. It may describe trends and suggest operational follow-up, but has no
field for a per-part decision and cannot modify the inspection workflow. The
default is a deterministic template:

```bash
curl "http://127.0.0.1:8000/v1/operations/briefing?period=24h"
```

In the Operations Dashboard, select a period and use **Generate briefing**.
The dashboard does not regenerate it during its regular data refresh, avoiding
an unnecessary local-model call every ten seconds.

To enable a local Ollama narrator, configure it separately from the
per-inspection explainer:

```bash
export OPERATIONS_BRIEFING_PROVIDER=ollama
export OPERATIONS_BRIEFING_OLLAMA_MODEL=qwen2.5:1.5b
uvicorn app.api:app --reload
```

The response is schema-validated. Every action's policy reason and every SOP
reference must occur in the supplied snapshot; malformed output, an invented
reference, an unexpected field such as `decision`, or an unavailable Ollama
service returns the deterministic template with a `fallback_reason`.

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

For policy-driven inspection outcomes, the additional
`app/procedure_routing.py` table maps a critical reason code to an approved SOP
**parent section**. Pinecone then searches only chunks bearing that document ID
and parent-section metadata. This prevents a generic section such as `Purpose`
from outranking an operational one such as `Required review steps`. Currently a
parent section maps to one chunk. A larger future SOP can have multiple child
chunks with the same `parent_section` and different `chunk_index` values; the
routing filter remains unchanged while vector search chooses the best child.

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

The test suite checks Pinecone request/response contracts with a fake index.
After every SOP indexing release, measure the *real* vector retrieval quality
against the live Pinecone namespace separately:

```bash
python scripts/evaluate_pinecone_retrieval.py
# Optional CI/release quality gates:
python scripts/evaluate_pinecone_retrieval.py \
  --min-recall-at-1 0.90 --min-section-accuracy-at-1 0.80
```

The benchmark prints the expected section and the returned top three citations
for each versioned query, then reports Recall@1, Recall@3, MRR and exact
section accuracy@1. Its four initial cases are a small regression set, not a
claim that every future SOP question is covered. Add a labelled case whenever a
reviewer observes a poor retrieval result.

The prior command intentionally measures unconstrained semantic search. Verify
the production inspection path separately after each Pinecone sync; this path
starts with a policy reason code and applies the SOP parent-section filter:

```bash
python scripts/evaluate_pinecone_policy_routing.py
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
