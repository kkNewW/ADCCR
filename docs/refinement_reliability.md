# Refinement reliability analysis

The evaluation pipeline supports a fixed confidence gate for local
refinement. Confidence is the geometric mean probability of the generated
coordinate tokens. With a threshold of `0.5`, a keypoint is refined when its
confidence is at least `0.5`; otherwise, its coarse coordinate is retained.
The normalized coarse error is used only for post-hoc stratification and is
never available to the inference-time gate.

Run the two policies and produce the reliability summary with:

```bash
bash scripts/refinement_reliability.sh
```

The command writes independent always-on and confidence-gated predictions,
then matches them by COCO annotation identifier. Only annotated keypoints are
included. For each keypoint, it records coarse, final, and ground-truth
coordinates in the resized `224 x 224` person-instance coordinate system.

The analysis uses

```text
q = L-infinity(coarse - ground_truth) / category-aware crop size
```

and assigns outcomes from the Euclidean error change with a `0.5 px`
tolerance:

- improved: final error - coarse error < -0.5 px;
- unchanged: absolute error change <= 0.5 px;
- worsened: final error - coarse error > 0.5 px.

Skipped predictions have identical coarse and final coordinates and are
therefore counted as unchanged. The generated JSON and CSV files include raw
counts as well as percentages so that all rounded values can be audited.
