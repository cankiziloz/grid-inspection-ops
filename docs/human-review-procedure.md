# Human review procedure

**Document ID:** SOP-REV-02  
**Status:** Portfolio demo procedure  
**Owner:** Quality Operations  
**Version:** 1.0

## Purpose

Describe how an operator resolves an `INSPECT` case. Operators do not change
the initial CV/policy decision; they add a final disposition with evidence and
a reason.

## Entry condition

Review begins only when an inspection record has:

```text
decision: INSPECT
review_status: PENDING
final_decision: null
```

## Required review steps

1. Open the pending record in the review dashboard.
2. Confirm the displayed `part_id`, model version, score, thresholds, and
   critical-policy reason code.
3. Compare the original image with the anomaly mask and highlighted regions.
4. Confirm that the image is usable; quality-gate failures require capture or
   integration investigation before a disposition is made.
5. Enter reviewer identity and a specific reason.
6. Resolve exactly once to `PASS` or `REJECT`.

## Resolution rules

| Resolution | Use when |
| --- | --- |
| `PASS` | The operator determines the highlighted variation is acceptable under the current procedure. |
| `REJECT` | The operator confirms a defect, cannot establish acceptability, or the process requires conservative disposition. |

The UI intentionally does not allow a reviewer to resolve an item back to
`INSPECT`. Escalate an unresolved case instead.

## Audit requirements

The service stores the immutable initial decision and the review resolution:

- reviewer ID;
- UTC review timestamp;
- final disposition;
- free-text reason;
- `HUMAN_REVIEW_RESOLVED` audit event.

This record supports later false-positive/false-negative analysis without
rewriting the model's original output.
