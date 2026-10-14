# num_function: exact-count symbolic replacement

This is a separate experiment on branch `num_function`, based on
`SFT_FINAL_RESEARCH` commit `28e30a3`. The existing maximum-term LOO method,
other scientific backends, and their published branches remain available.

`--num-functions N` means **exactly N distinct function terms**, not a maximum
and not N transformer blocks. The compatibility alias `--symbolic-max-terms N`
has exact-count semantics in this backend only. Prefer the explicit new flag.
Counts 1..9 are supported; the requested grid uses 3, 5, 7 and 9.

Library: linear, clipped square, clipped cube, tanh, sin, constant, GELU, SiLU,
Softsign. Functions are added as a weighted sum, so reordering them does not
create a new expression. Search combinations rather than duplicate permutations:

| Exact count | Combinations considered |
| --- | ---: |
| 3 | 84 |
| 5 | 126 |
| 7 | 36 |
| 9 | 1 |

For each combination, solve per-latent-coordinate coefficients by pseudoinverse
on fitting training images; evaluate full-output reconstruction on held-out
training images; choose the smallest penalized error and refit coefficients on
available calibration tokens. The term-count penalty is constant within an
exact-count search and therefore does not change its ranking. Normal training
then updates projections/coefficients/PEFT/LayerNorms/head. Selected function
identities stay fixed. Some coefficients can become small; exact selection does
not guarantee every function contributes strongly or that accuracy improves.

LOO screening remains independent of function count: classifier-only warm-up,
training-only signed deletion loss, minimum-delta position for replacement,
maximum surviving delta for single-block PEFT. All other pretrained weights are
frozen, with remaining LayerNorms and the task head trainable. Existing dataset
split/proxy behavior is preserved rather than claimed to reproduce exact VTAB.

From the full repository root:

```bash
python3 run.py num_function --dataset pets --num-functions 3 \
  --filter-rank 128 --adapter-type lora --lora-rank 8 --lora-alpha 16 \
  --loo-samples 512 --head-warmup-epochs 5 --epochs -1 --device cuda
```

Other supported targeted methods: DoRA, PaCA, RPaCA and their LoRA/DoRA fused
variants; no-PEFT is available. Uni-LoRA and Uni-DoRA are excluded. The supplied
192-task grid uses LoRA by default; `--adapter-type dora` selects a DoRA-only grid.

Artifacts: `initial_filter_equations.json` and printed selected names before
recovery training; `filter_equations.json`, `metrics_summary.json`, `best_model.pt`,
history/plots after training. Reports store exact requested/selected counts and
function names. A saved checkpoint is evaluated with:

```bash
python3 run.py evaluate-num-function --summary /absolute/run/metrics_summary.json --device cuda
```

For the standalone A100 scheduler provided in chat, extract the **whole branch
repository** under `/export/home/achyut/Sarvesh/SFT_NUM_FUNCTION`, with the backend
at `SFT_NUM_FUNCTION/train_num_function.py` beneath that root. Alternatively pass
`--repo-root /your/full/repository/root`. The scheduler is independent of
`research_common.sh`, reuses `/export/home/achyut/Sarvesh/SFT_BEST/data`, and uses
locked dataset/backbone preparation from the repository before concurrent runs.
Use fresh campaign names. Early stopping is default; `--epochs 100` is fixed.
