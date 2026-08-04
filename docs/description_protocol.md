# Description protocol

## Scope

The structured training bank contains the 17 COCO
keypoints. Each keypoint contains manually prepared name,
anatomy, relation and visual fields. The additional
resources contain canonical descriptions and localization
question templates for 24 keypoints.

Only query composition and mode sampling are automatic.
The semantic fields themselves are not automatically
generated.

## Quality control

Each entry is checked using the following criteria:

1. consistency with the dataset keypoint definition;
2. anatomical plausibility;
3. correct left-right laterality;
4. consistency with adjacent and symmetric landmarks;
5. absence of ambiguous coordinate supervision;
6. avoidance of overly view-dependent visual cues;
7. grammatical clarity and consistent terminology.

An entry that fails any check is revised and checked again
before release. The resource validator is run after every
change.

## Adding a training-supervised keypoint

1. Add a unique canonical keypoint name.
2. Provide non-empty name, anatomy, relation and visual
   fields.
3. Add the keypoint to the training annotation mapping.
4. Provide a category-aware crop size if local refinement
   is enabled.
5. Provide canonical localization descriptions and
   question templates.
6. Run the resource validator and relevant unit tests.

## Adding a query-only keypoint

1. Mark the keypoint explicitly as query-only.
2. Provide its canonical description and inference
   templates.
3. Define its evaluation annotation and metric.
4. Do not add it to the training keypoint list, labels or
   coordinate supervision.
5. Provide a crop size only when local refinement is used.
6. Run the resource validator before evaluation.