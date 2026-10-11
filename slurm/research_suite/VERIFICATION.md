# Research launcher verification

The root suite passed **83 tests**, including **21 new research-suite checks**.
The 21 checks passed again after the final compiler-figure and submission-record
updates. They verify:

- All 7,140 unique dataset–method–block tasks per seed, all seven groups,
  Caltech101 exclusion, no SFT-only dense+PEFT entries and no Uni entries in LOO.
- Every distinct backend/configuration command against the actual original,
  symbolic, LOO and subtraction CLI parsers; LOO/subtraction also run their
  argument validators. No training or pretrained download is triggered.
- Both requested orthogonality coefficients, valid direct/fused/Uni flags,
  highest-correlation cutoff −1 and deferred removal-candidate test evaluation.
- Correct large-array offsets, complete task coverage and preservation of job
  records across separately submitted groups of one campaign.
- Slurm shell syntax and locating helper files from a spooled shell script.
- A synthetic complete block sweep whose highest test score is at a different
  position than the highest validation score: only the validation winner is
  selected. An incomplete sweep produces no winner.
- Synthetic removal evaluation with a stub evaluator: by default only the
  validation winner gets tested; the optional all-removals setting tests all
  candidates. Missing test values remain missing before evaluation.
- Saved-result CSV compilation, task coverage, PNG/SVG/PDF figures, local HTML
  index and preserving all twelve LOO candidate indices separately from the
  selected replacement index.

The four backend directories have no changes relative to `0cca0fa`. Their
previous **637 scientific checks** remain applicable to those identical files;
they were not rerun for this launcher-only change. Full-scale dataset training,
cluster software compatibility, Slurm limits, runtime and accuracy remain
unverified. These launchers preserve existing code behavior and do not establish
exact reproduction of the paper.
