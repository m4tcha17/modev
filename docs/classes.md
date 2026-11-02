# Copra Grades

Source: `model-development-instructions.md` §1. Each photo's `copra_class` (`A`–`F`) is the partner buying station's own moisture grade, assigned by the owner's pasa grading of the whole copra. Displayed as "Grade A" … "Grade F". The label is read straight from the CSV, never derived from a moisture reading.

| Grade | Moisture content | Note |
|---|---|---|
| F | below 6% | Penalized |
| A | 6% to <9% | |
| B | 9% to <12% | |
| C | 12% to <15% | |
| D | 15% to <18% | |
| E | 18% and above | Lowest payout / quality |

**Boundary rule:** each band includes its lower edge and excludes its upper edge. A value on 9, 12, 15 or 18% belongs to the higher grade. Unit is moisture content (%).

Ordered by moisture: **F < A < B < C < D < E**. The letters are not in moisture order, so F (driest) sits next to A, not next to E.

The scheme comes from the buying station owner, not PCA and not a publication; do not cite a source for it. Do not use other grade names (Resecada, Bodega, Semi-Resecada, Corriente, Non-Merchantable, PCA Class 1/2/3) anywhere.

In code: `dataset.CLASS_DESCRIPTIONS` and `dataset.MOISTURE_ORDER`.

## What this means for the pipeline

- **Ordered grades.** Mixing up neighbouring bands (A↔B) is a smaller mistake than distant ones (F↔E), but Macro F1 and accuracy treat every mistake the same. Show the confusion matrix in `MOISTURE_ORDER` so near-miss errors sit next to the diagonal.
- **Label check (acceptance L1):** one pooled Brown-Duvel reading per grade must fall inside its band, using the boundary rule above. A miss is reported as a label limitation; images are never relabelled.
- **Color signal.** Drier copra shifts from pale/white toward tan/brown, which the HSV/LAB features pick up.
