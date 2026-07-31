# end-of-outbreak-vbd

Python code accompanying "End-of-outbreak determination under seasonal transmission" by
Hart *et al.*

## Installation

We recommend using the [pixi](https://pixi.prefix.dev/) package manager. Once pixi is
installed, running

```bash
pixi install --locked
```

creates a local environment (under `.pixi/`) containing the exact dependency versions 
recorded in `pixi.lock`, which were used to generate the results in the paper. Prefixing
a command with `pixi run` executes it in that environment.

## Reproducing the results

The results and figures in the paper can be reproduced by running the scripts in the
[`scripts`](scripts/) directory with `pixi run python`. For example, to reproduce the
results and panels in Figure 4, run:

```bash
pixi run python scripts/lazio_outbreak.py
```

Alternatively, the full pipeline for all results and figures can be run with
[Snakemake](https://snakemake.readthedocs.io/):

```bash
pixi run snakemake --cores 1 --forceall
```
