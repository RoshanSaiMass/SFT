# Final research source download

`SFT-final-res.zip` is the current source bundle for branch `sft_final_res`.
It includes the four retained backends, common CLI, launchers, examples and tests.
The dedicated `SFT_SNIP_COMPACT` experiment and its download bundles are excluded.
Datasets, cached weights, checkpoints, generated results and Git metadata are
not included.

- Source commit: `7a215e1d321c4e4ac79e883c93ebb6db1565fde2`
- SHA256: `979f5c0ad9764e8604b55f38de8a927e18df67bbffeef6c606cd75093bf9ec59`
- Size: 549,530 bytes

The archive passed its CRC check, extracted research-grid count (7,140 tasks),
master compiler help, shell syntax and chunked-submission dry run. The current
root suite passed 83 CPU checks. Its four unchanged scientific backends passed
637 checks in the preceding verification; these were not rerun for this
launcher-only change. Cluster training and dataset downloads were not exercised.
The source commit precedes the archive publication commit because the bundle
excludes this downloads directory.

## Standalone research Slurm scripts

`SFT-research-slurm-suite.zip` contains the seven separate array launchers,
configuration file, submission helper and master CSV/plot compiler. It calls
your existing `SFT_BEST`, `SFT_FINAL`, `SFT_EXP`, `SFT_LOO` and `SFT_COEFF_RED`
cluster folders; it does not contain or overwrite their scientific code.

- Source commit: `7a215e1d321c4e4ac79e883c93ebb6db1565fde2`
- SHA256: `af9bbf0ca742c577ba32510755bd142f592bd2642c2033f0d8f31935b4c15361`
- Size: 37,773 bytes

Extract the whole bundle, edit `slurm/research_suite/config.sh` if needed, and
run `bash slurm/research_suite/submit_suite.sh` from its root. The README lists
commands for each individual experiment and the selectable epoch budget.
The standalone archive passed CRC, extracted grid count, master CLI help, shell
syntax and chunked-submission preview checks as well.

Open the ZIP on this branch and use GitHub's download button, or select
**Code → Download ZIP** for `sft_final_res`.
