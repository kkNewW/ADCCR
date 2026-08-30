# Refinement reliability analysis

The evaluation pipeline supports a fixed confidence gate for local
refinement. Confidence is the geometric mean probability of the generated
coordinate tokens. With a threshold of `0.5`, a keypoint is refined when its
confidence is at least `0.5`; otherwise, its coarse coordinate is retained.
The normalized coarse error is used only for post-hoc stratification and is
never available to the inference-time gate.

Run the always-on validation stage, which now writes raw JSONL directly, and
then produce both policy summaries from the same refinement candidates:

```bash
python utils/run_config.py --config configs/coco_full.json \
  --stage eval_refinement_always_on
python utils/run_config.py --config configs/coco_full.json \
  --stage analyze_refinement_reliability
```

The validation command writes
`results/refinement_reliability/raw_refinement_predictions.jsonl` with one row
per visible keypoint. Each row stores the keypoint-specific description, crop
size, coarse coordinate, always-on refinement candidate, ground truth, and
generated-token confidence in the resized `224 x 224` person-instance system.
The analyzer applies the fixed gate offline to those same candidates, so the
always-on and gated policies are paired exactly rather than matched between
two independently generated files.

The analysis uses

```text
q = L-infinity(coarse - ground_truth) / category-aware crop size
```

The infinity norm is used only for stratification. Euclidean pixel errors are
still used for the mean-error and outcome columns. The four reporting strata
are the same as Table 16: `0 <= q <= 0.10`,
`0.10 < q <= 0.25`, `0.25 < q <= 0.50`, and `q > 0.50`.

and assigns outcomes from the Euclidean error change with a `0.5 px`
tolerance:

- improved: final error - coarse error < -0.5 px;
- unchanged: absolute error change <= 0.5 px;
- worsened: final error - coarse error > 0.5 px.

Skipped predictions have identical coarse and final coordinates and are
therefore counted as unchanged. The generated JSON and CSV files include raw
counts as well as percentages so that all rounded values can be audited.
