# Reject and quarantine policy

**Document ID:** SOP-QA-04  
**Status:** Portfolio demo procedure  
**Owner:** Quality Operations  
**Version:** 1.0

## Scope

Apply this procedure to an automatic `REJECT` or a human-confirmed final
`REJECT` in Grid Inspection Ops.

## Immediate actions

1. Prevent the part from continuing through the simulated release path.
2. Retain the original image and generated anomaly mask with the inspection
   record.
3. Record `part_id`, inspection ID, model version, score, thresholds, and
   critical-policy reason code.
4. Verify the highlighted regions if local process rules require a secondary
   visual check.
5. Review recent rejects from the same station/camera before resuming normal
   operation.

## Non-overridable evidence

The following must never be removed when documenting a reject:

- initial automated decision;
- final decision;
- reviewer information if a review occurred;
- image/mask references;
- policy reason codes;
- model and policy versions.

## Portfolio boundary

“Quarantine” represents a workflow state in this project. No physical machine
or part-routing control is performed by the code.
