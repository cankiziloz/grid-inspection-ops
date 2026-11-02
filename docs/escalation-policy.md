# Escalation policy

**Document ID:** SOP-ESC-03  
**Status:** Portfolio demo procedure  
**Owner:** Quality Operations  
**Version:** 1.0

## Purpose

Define when a human review should be escalated rather than resolved locally.

## Escalate when

- a reviewer cannot determine `PASS` or `REJECT` from available evidence;
- repeated `INSPECT` outcomes indicate camera, lighting, or alignment drift;
- reject volume from one station/camera is unusually high;
- an anomaly category appears new or materially different from known cases;
- image quality gates repeatedly fail;
- the model artifact, checksum, or policy configuration cannot be verified.

## Escalation record

The escalation should reference the inspection ID and include:

- part, station, and camera context when available;
- image and mask references;
- policy reason code;
- reviewer observations;
- requested owner: quality, maintenance, or platform engineering.

## Resolution principle

Until escalation is resolved, the system must not turn an unresolved `INSPECT`
into an automatic `PASS`. The safe action is a review hold.
