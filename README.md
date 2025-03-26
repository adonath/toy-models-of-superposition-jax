# Toy Models of Superposition: Replication in JAX

This repository contains a replication of Anthropic's "Toy Models of Superposition" in JAX [1, 2].

I made the JAX implementation a bit more scalable and flexible than a single notebook. This includes:

- Single entry `make.py` file to run the toy experiments and plotting figures
- Experiments can be configured in `.toml` files contaiend in `config`
- Running the experiments and plotting the results are separate steps
- The fitted toy models are store in `safetensors` format in the `results` folder


## Getting Started

I would recommeend to use `uv`, which will automatically detect the `uv.lock` file and create and reproducible environment:

```bash
uv run make.py
```

```bash
uv run make.py plot --config configs/varying-sparsity.toml --plot-type superposition
```

## Results

I can basically exactly(!) reproduce the figures shown in [1]

### Intro Figure

![intro-figure](images/features-5-hidden-2-instances-10-relu-intro.png)

### Superposition

![superposition](images/features-20-hidden-5-instances-7-relu-superposition.png)


## References

_[1]: https://transformer-circuits.pub/2022/toy_model/index.html
_[2]: https://github.com/anthropics/toy-models-of-superposition/tree/main
