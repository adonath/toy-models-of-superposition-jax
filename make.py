import logging
import tomllib
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import click
import einops
import jax
import optax
from jax import numpy as jnp
from jax.tree_util import register_dataclass
from safetensors import safe_open
from safetensors.flax import save_file
from tqdm.autonotebook import trange

from plot import plot_demonstrate_superposition, plot_intro_diagram

log = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

PATH_BASE = Path(__file__).parent
PATH_RESULTS = PATH_BASE / "results"


def define_registry(registry):
    """Define registry"""
    # allow reverse "two way" access of items
    for key, item in list(registry.items()):
        registry[item] = key

    def get_with_error(key):
        if key not in registry:
            message = f"{key} is not a valid key, choose from {list(registry.keys())}"
            raise KeyError(message)

        return registry[key]

    return get_with_error


ACTIVATION = {"relu": jax.nn.relu, "gelu": jax.nn.gelu, "identity": lambda x: x}

get_activation = define_registry(ACTIVATION)

INIT = {
    "xavier-normal": jax.nn.initializers.glorot_normal(in_axis=-2, out_axis=-1),
    "zeros": lambda key, shape, dtype: jnp.zeros(shape, dtype=dtype),
}

get_init = define_registry(INIT)


def validate_key(data, key, validator):
    """Validate a dict entry by applying a validator"""
    if key in data:
        data[key] = validator(data[key])


def validate_array(value, n):
    """Validate array"""
    if isinstance(value, dict):
        return value["lambda"] ** -jnp.linspace(0, 1, n)

    return jnp.array(value)


@dataclass
class Config:
    """Toy model config"""

    n_features: int
    n_hidden: int
    n_instances: int
    activation: str = "relu"
    w_init: str = "xavier-normal"
    b_init: str = "zeros"
    seed: int = 923836
    device: str = "cpu"
    dtype: jnp.dtype = jnp.float32
    feature_probability: jax.Array = field(default_factory=lambda: jnp.ones(1))
    feature_importance: jax.Array = field(default_factory=lambda: jnp.ones(1))

    def __post_init__(self):
        if len(self.feature_importance) not in [1, self.n_features]:
            message = (
                f"Feature importance requires length of 1 or {self.n_features}, "
                f"got length of {len(self.feature_importance)}"
            )

            raise ValueError(message)

        if len(self.feature_probability) not in [1, self.n_instances]:
            message = (
                f"Feature probability requires length of 1 or {self.n_instances}, "
                f"got length of {len(self.feature_probability)}"
            )

            raise ValueError(message)

    def key(self):
        """JAX random key"""
        return jax.random.key(self.seed)

    @property
    def result_filename(self):
        """Result filename for a given config"""
        return f"features-{self.n_features}-hidden-{self.n_hidden}-instances-{self.n_instances}-{self.activation}"

    @classmethod
    def read(cls, filename):
        """Read config from a TOML file"""
        log.info(f"Reading {filename}")

        with open(filename, "rb") as f:
            data = tomllib.load(f)

        validate_key(data, "dtype", lambda _: getattr(jnp, _))
        validate_key(
            data, "feature_probability", partial(validate_array, n=data["n_instances"])
        )
        validate_key(
            data, "feature_importance", partial(validate_array, n=data["n_features"])
        )
        return cls(**data)


@partial(register_dataclass, data_fields=("w", "b_final"), meta_fields=("activation",))
@dataclass
class Model:
    """Model"""

    w: jax.Array
    b_final: jax.Array
    activation: callable = field(default=jax.nn.relu, metadata={"static": True})

    @property
    def n_instance(self):
        """Number of instances"""
        return self.w.shape[0]

    @classmethod
    def from_config(cls, config):
        """Create from config"""
        shape = (config.n_instances, config.n_features, config.n_hidden)
        w = get_init(config.w_init)(key=config.key(), shape=shape, dtype=config.dtype)

        shape = (config.n_instances, config.n_features)
        b_final = get_init(config.b_init)(
            key=config.key(), shape=shape, dtype=config.dtype
        )

        activation = get_activation(config.activation)
        return cls(w=w, b_final=b_final, activation=activation)

    def __call__(self, x):
        # TODO: check whether regular matmul improves performance
        hidden = jnp.einsum("...if,ifh->...ih", x, self.w)
        out = jnp.einsum("...ih,ifh->...if", hidden, self.w)
        out = out + self.b_final
        out = self.activation(out)
        return out

    @classmethod
    def read(cls, filename, device=None):
        """Read model from safetensors file"""

        log.info(f"Reading {filename}")

        with safe_open(filename, framework="flax", device=device) as f:
            # reorder according to metadata, which maps index to key / path
            data = {key: f.get_tensor(key) for key in f.keys()}
            data["activation"] = get_activation(f.metadata()["activation"])

        data.pop("losses")
        return cls(**data)

    def write(self, filename, extra=None, meta_extra=None):
        """Write model to safetensors file"""
        extra = {} if extra is None else extra
        meta_extra = {} if meta_extra is None else meta_extra

        data = {"w": self.w, "b_final": self.b_final}
        data.update(extra)

        metadata = {"activation": get_activation(self.activation)}
        metadata.update(meta_extra)

        log.info(f"Writing {filename}")
        save_file(data, filename, metadata=metadata)


@dataclass
class DataGenerator:
    """Data generator"""

    n_features: int
    feature_probability: jax.Array
    feature_importance: jax.Array
    batch_size: int
    key: jax.Array
    device: str = "cpu"

    @classmethod
    def from_config(cls, config):
        """Create from config object"""
        probability = jnp.expand_dims(config.feature_probability, axis=(0, 2))
        importance = jnp.expand_dims(config.feature_importance, axis=(0, 1))

        return cls(
            n_features=config.n_features,
            feature_probability=probability,
            feature_importance=importance,
            batch_size=1024,
            key=config.key(),
            device=config.device,
        )

    def __iter__(self):
        key = self.key

        while True:
            n_instances = self.feature_probability.shape[1]
            shape = (self.batch_size, n_instances, self.n_features)

            key, subkey = jax.random.split(key)
            features = jax.random.uniform(key=subkey, shape=shape)

            key, subkey = jax.random.split(key)
            helper = jax.random.uniform(key=subkey, shape=features.shape)
            features = jnp.where(helper <= self.feature_probability, features, 0)
            yield features


def loss_fn(model, x, importance):
    """Calculate weighted MSE loss"""
    error = importance * (jnp.abs(x) - model(x)) ** 2
    loss = einops.reduce(error, "b i f -> i", "mean").sum()
    return loss


def optimize(model, data_generator, n_steps=10_000, print_freq=100, learning_rate=1e-3):
    """Optimize model"""
    # Initialize optimizer
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(model)

    importance = data_generator.feature_importance

    @jax.jit
    def train_step(state, batch):
        """Single training step"""
        model, opt_state = state

        loss, grads = jax.value_and_grad(loss_fn)(model, batch, importance)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = optax.apply_updates(model, updates)

        return (new_model, new_opt_state), loss

    # Initialize state
    state = (model, opt_state)

    # Training loop
    data_iter = iter(data_generator)
    losses = []

    with trange(n_steps) as t:
        for step in t:
            batch = next(data_iter)
            state, loss = train_step(state, batch)
            losses.append(loss)

            if step % print_freq == 0 or (step + 1 == n_steps):
                t.set_postfix(
                    loss=loss.item() / model.n_instance,
                )

    trained_model = state[0]
    return trained_model, jnp.array(losses)


@click.group()
def cli():
    pass


@cli.command("train")
@click.option(
    "--config",
    type=click.Path(),
    help="Path to config TOML file",
    default=PATH_BASE / "configs/default.toml",
)
@click.option(
    "--learning-rate", type=float, help="Learning rate for optimization", default=1e-3
)
@click.option(
    "--n-steps", type=int, help="Number of optimization steps", default=10_000
)
@click.option(
    "--print-freq", type=int, help="Frequency of printing progress", default=100
)
def cli_train(config, learning_rate, n_steps, print_freq):
    """Train a toy model"""
    config = Config.read(config)

    model = Model.from_config(config=config)
    data_generator = DataGenerator.from_config(config=config)

    kwargs = {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "print_freq": print_freq,
    }

    model_optimized, losses = optimize(
        model=model, data_generator=data_generator, **kwargs
    )

    kwargs["config"] = config
    # Save the optimized model
    output_path = PATH_RESULTS / f"{config.result_filename}.safetensors"
    model_optimized.write(
        output_path,
        extra={"losses": losses},
        meta_extra={key: str(value) for key, value in kwargs.items()},
    )


PLOT_TYPES = {
    "intro": plot_intro_diagram,
    "superposition": plot_demonstrate_superposition,
}


@cli.command("plot")
@click.option(
    "--config",
    type=click.Path(),
    help="Path to config TOML file",
    default=PATH_BASE / "configs/default.toml",
)
@click.option(
    "--plot-type",
    type=click.Choice(PLOT_TYPES.keys()),
    help="Choose plot type",
    default="intro",
)
def cli_plot(config, plot_type):
    """Make plots from trained models"""
    config = Config.read(config)
    model = Model.read(PATH_RESULTS / f"{config.result_filename}.safetensors")
    output_path = PATH_RESULTS / f"{config.result_filename}-{plot_type}.png"

    PLOT_TYPES[plot_type](model=model, config=config, filename=output_path)


if __name__ == "__main__":
    cli()
