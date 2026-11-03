# Automated image-ingestion contract

**Document ID:** SOP-ING-01  
**Status:** Portfolio demo procedure  
**Owner:** Inspection Platform  
**Version:** 1.0

## Purpose

Define the boundary between an automated camera/edge system and the Grid
Inspection Ops API. An operator does not manually take or upload an image in
the intended production workflow.

## Intended flow

```text
Camera or edge collector
  -> capture validation
  -> idempotent API submission
  -> Grid Inspection Ops
  -> PASS / INSPECT / REJECT event
```

## Required capture metadata

A production gateway should attach the following immutable fields to every
capture. The portfolio edge endpoint (`POST /v1/edge/inspections`) validates
all of them, including the image checksum. They are supplied by an automated
gateway or simulator, never fabricated by a dashboard operator.

| Field | Example | Purpose |
| --- | --- | --- |
| `part_id` | `grid-2026-000184` | Correlates result with the manufactured part. |
| `station_id` | `line-2` | Limits a result to the correct process context. |
| `camera_id` | `line-2-camera-1` | Supports camera-health and drift analysis. |
| `captured_at` | ISO-8601 UTC timestamp | Enables ordered audit and incident analysis. |
| `frame_id` | camera-generated unique ID | Enables idempotent retry handling. |
| `image_sha256` | SHA-256 digest | Detects accidental image corruption or duplication. |

The API treats `frame_id` as an idempotency key. Replaying the same frame
returns the original inspection rather than creating a second record.

## Image acceptance rules

- The gateway accepts only PNG or JPEG images.
- The image must be associated with one `part_id` and one `frame_id`.
- A timeout may be retried with the same `frame_id`; it must not create a
  second inspection record.
- Invalid content type, checksum mismatch, or duplicate frame with conflicting
  content is routed to the integration error channel, not silently inspected.
- The inspection service separately checks resolution, exposure, contrast,
  sharpness, and STN valid coverage. A failed quality gate yields `INSPECT`,
  never automatic `PASS`.

## Response handling

| Initial decision | Automated downstream action |
| --- | --- |
| `PASS` | Record the result and release the part according to the local process. |
| `INSPECT` | Place the part in a review hold and create a human-review work item. |
| `REJECT` | Apply the quarantine policy in `reject-and-quarantine-policy.md`. |

## Portfolio boundary

This repository does not connect to a PLC, physical camera, MES, or message
broker. The dashboard file picker exists only to demonstrate the API with test
images. A real deployment would replace it with an authenticated edge gateway
or message consumer.
