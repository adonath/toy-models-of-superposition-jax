# Toy Models of Superposition: Replication in JAX

This repository contains a replication of Anthropic's "Toy Models of Superposition" in JAX.

The original code can be found here: https://github.com/anthropics/toy-models-of-superposition/tree/main

And the article is here: https://transformer-circuits.pub/2022/toy_model/index.html


## Getting Started

I would recommeend to use `uv`, which will automatically detect the `uv.lock` file and create and reproducible environment:

```bash
uv run make.py
```

```bash
uv run make.py plot --config configs/varying-sparsity.toml --plot-type superposition
```

## Results

### Intro Figure

![intro-figure](images/features-5-hidden-2-instances-10-relu-intro.png)

### Superposition

![superposition](images/features-20-hidden-5-instances-7-relu-superposition.png)
