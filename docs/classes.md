# Copra Classes

Each photo's `copra_class` (A–F) stands for a moisture-content band. The label is still read straight from the CSV; these bands explain what it means.

| Class | Moisture content | Meaning |
|---|---|---|
| F | below 6% | Penalty (over-dried) |
| A | 6–9% | |
| B | 9–12% | |
| C | 12–15% | |
| D | 15–18% | |
| E | 18% and above | Lowest payout / quality |

Ordered by moisture: **F < A < B < C < D < E**. The letters are not in moisture order, so F (driest) sits next to A, not next to E.

In code: `dataset.CLASS_DESCRIPTIONS` and `dataset.MOISTURE_ORDER`.

## What this means for the pipeline

- **Ordered classes.** Mixing up neighbouring bands (A↔B) is a smaller mistake than mixing up distant ones (F↔E). Macro F1 and accuracy treat every mistake the same. Show the confusion matrix with rows and columns in `MOISTURE_ORDER` so near-miss errors sit next to the diagonal.
- **Errors near cutoffs.** Photos with moisture close to 6/9/12/15/18% are the most likely to be labelled one band off. Expect most confusion between neighbouring classes.
- **Color signal.** Drier copra shifts from pale/white toward tan/brown, which is what the HSV/LAB features pick up.

## Open

- Boundary rule: is exactly 9.0% class A or B? (Same for 6, 12, 15, 18.) Doesn't affect training, since labels come from the CSV, but the docs and app text should state it.
- Unit assumed to be moisture content %. Confirm.
