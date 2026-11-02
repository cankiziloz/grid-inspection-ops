# Operational documents

These documents are fictional, version-controlled procedures for the Grid
Inspection Ops portfolio project. They demonstrate how an inspection workflow
could be governed; they are not instructions for a real production line.

The document set deliberately separates two responsibilities:

- The camera/edge integration automatically submits inspection images.
- A human operator participates only when the deterministic policy returns
  `INSPECT`.

The current web upload form is a local demo/test harness. It is not the
intended production operator workflow.

| Document | Purpose |
| --- | --- |
| `automated-image-ingestion-contract.md` | Camera/edge-to-API contract and reliability rules. |
| `human-review-procedure.md` | How an operator resolves `INSPECT` cases. |
| `reject-and-quarantine-policy.md` | Immediate handling of automatic or human-confirmed rejects. |
| `escalation-policy.md` | When a supervisor or quality owner must be involved. |

In a later RAG phase, only these approved documents (or explicitly approved
replacements) will be indexed. An LLM may explain their relevant clauses but
cannot change the CV or critical-policy decision.

## Tracked retrieval hardening

**RAG-EVAL-01 — Retrieval evaluation benchmark** is a required follow-up before
declaring the RAG layer production-ready. It will add a versioned set of
keyword and natural-language queries, each labelled with an expected SOP and
section, and will measure Recall@1, Recall@3, MRR, and section accuracy. The
current tests validate deterministic policy-reason routing; they do not claim
that arbitrary free-text queries are always retrieved correctly.
