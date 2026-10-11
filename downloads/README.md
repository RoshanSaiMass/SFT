# Final research source download

`SFT-final-res.zip` is the current source bundle for branch `sft_final_res`.
It includes the four retained backends, common CLI, launchers, examples and tests.
The dedicated `SFT_SNIP_COMPACT` experiment and its download bundles are excluded.
Datasets, cached weights, checkpoints, generated results and Git metadata are
not included.

- Source commit: `0083fa98eeb0235932bed39a98d1e8093cc079f9`
- SHA256: `4b76e4145e6fdf16081e5b749089f79f228dffa2bba4cb76afc5e719e2c996c4`
- Size: 518,054 bytes

The archive passed its CRC check, seven extracted CLI help checks, and the
extracted original-grid dry run with both training budgets (520 tasks).
The dedicated SNIP workflow was rejected as expected. The source passed 699
CPU tests; cluster training and dataset downloads were not exercised here.
The source commit precedes the archive publication commit because the bundle
excludes this downloads directory.

Open the ZIP on this branch and use GitHub's download button, or select
**Code → Download ZIP** for `sft_final_res`.
