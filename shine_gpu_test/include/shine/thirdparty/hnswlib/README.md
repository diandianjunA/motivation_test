Vendored `hnswlib` headers used by SHINE's offline index builder.

Source:
- Repository: https://github.com/nmslib/hnswlib
- Imported commit: `c1b9b79af3d10c6ee7b5d0afa1ce851ae975254c`

This directory intentionally keeps only:
- `LICENSE`
- the `hnswlib/*.h` headers required to build `shine_offline_builder`

It is tracked as normal source files in this repository. No extra `git submodule` or separate clone step is needed.
