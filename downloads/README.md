# Downloadable SNIP compact source

`SFT-snip-compact.zip` contains the source tree at commit
`e780d0f773ebff5ad7eee679342fe08f13d75950`, with top-level directory `SFT-snip-compact/`.

SHA256: `5202280909059b6381272d84fa7b6f3ce814a6a6a1e962bc28aa61a824ceb9ec`

This includes the new `SFT_SNIP_COMPACT` workflow, common CLI, tests and unchanged
existing backends. It excludes the downloads directory, datasets, pretrained
weights, generated results and Git metadata. The archive passed its CRC check;
the extracted CLI was checked for both new training and evaluation entry points.

Open the ZIP on GitHub branch `sft-snip-compact` and use the download button, or
choose Code / Download ZIP for that branch. Use `run.py snip-compact` to run the
new method. It also supports `--filter-type dense --adapter-type none` for
the traditional FC filter baseline without PEFT. This version uses original-ViT SNIP for PEFT targeting and
separate paper-style candidate-network SNIP for compact placement. The older `SFT-integrated.zip` remains available for the previous
integrated codebase; it does not include this new workflow.

---

# Downloadable integrated source

`SFT-integrated.zip` contains the source tree at commit
`ca72920`, with top-level directory `SFT-integrated/`.

SHA256:
`5e17c940306d56628f00d7d094fc3500d80b95b3a55e36803cc518fea64d9a27`

The ZIP was integrity checked and extracted locally; the extracted common CLI
resolved its symbolic backend and shared paths correctly. It includes all three
workflows and excludes datasets, weights, checkpoints, generated results and
Git metadata. It does not include this downloads directory or the ZIP itself.

Open the ZIP file in GitHub and use the download button. Alternatively use
Code / Download ZIP on the `sft-integrated` branch.
