# %% [markdown] 
# # Toy Models of Superposition
# This notebook is a replication of https://github.com/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb in JAX.
# %%
from dataclasses import dataclass, field
from enum import Enum
from functools import partial

import einops
import jax
import optax
from jax import numpy as jnp
from jax.tree_util import register_dataclass
from tqdm.autonotebook import trange


def define_registry(registry):
    """Define registry"""

    def get_with_error(key):
        if key not in registry:
            message = f"{key} is not a valid key, choose from {list(registry.keys())}"
            raise KeyError(message)
        
        return registry[key]
        
    return get_with_error
    

ACTIVATION = {
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu
}

get_activation = define_registry(ACTIVATION)

INIT = {
    "xavier-normal": jax.nn.initializers.glorot_normal(),
    "zeros": lambda key, shape, dtype: jnp.zeros(shape, dtype=dtype),
}

get_init = define_registry(INIT)

class Axis(int, Enum):
    """Axis convention"""
    batch = 0
    instance = 1
    hidden = 2
    feature = 3



@dataclass
class Config:
    """Toy model config"""
    n_features: int
    n_hidden: int
    n_instances: int
    activation: str = "relu"
    w_init: str = "xavier-normal"
    b_init: str = "zeros"
    seed: int = 87234
    device: str = "cpu"
    dtype: jnp.dtype = jnp.float32

    def key(self):
        """JAX random key"""
        return jax.random.key(self.seed)


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
        return self.w.shape[Axis.instance]

    @classmethod
    def from_config(cls, config):
        """Create from config"""
        shape = (config.n_instances, config.n_features, config.n_hidden)
        w = get_init(config.w_init)(key=config.key(), shape=shape, dtype=config.dtype)

        shape = (config.n_instances, config.n_features)
        b_final = get_init(config.b_init)(key=config.key(), shape=shape, dtype=config.dtype)

        activation = get_activation(config.activation)
        return cls(w=w, b_final=b_final, activation=activation)

    def __call__(self, x):
        print(x.shape)
        hidden = jnp.einsum("...if,ifh->...ih", x, self.w)
        out = jnp.einsum("...ih,ifh->...if", hidden, self.w)
        out = out + self.b_final
        out = self.activation(out)
        return out


@dataclass
class DataGenerator:
    """Data generator"""
    n_features: int
    feature_probability: jax.Array
    feature_importance: jax.Array
    batch_size: int
    key: jax.Array
    device: str = "cpu"

    def __iter__(self):
        key = self.key
        
        while True:
            n_instances = self.feature_probability.shape[Axis.instance]
            shape = (self.batch_size, n_instances, self.n_features)
            
            key, subkey = jax.random.split(key)
            features = jax.random.uniform(key=subkey, shape=shape)

            key, subkey = jax.random.split(key)
            sparsify = jax.random.bernoulli(key=subkey, p=self.feature_probability, shape=shape)

            features = features.at[sparsify].set(0)
            yield features


def loss_fn(model, x, importance):
    """Calculate MSE loss"""
    y_pred = model(x)
    error = (importance * (jnp.abs(x) - y_pred)**2)
    loss = einops.reduce(error, 'b i f -> i', 'mean').sum()
    return loss
    #return jnp.mean(importance * (y_pred - x) ** 2, axis=(Axis.batch, Axis.hidden, Axis.feature)).sum()


def optimize(model, data_generator, steps=10_000, print_freq=100, learning_rate=1e-3):
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

    with trange(steps) as t:
        for step in t:
            batch = next(data_iter)
            state, loss = train_step(state, batch)
            losses.append(loss)
            
            if step % print_freq == 0 or (step + 1 == steps):
                t.set_postfix(
                    loss=loss.item() / model.n_instance,
                )
    
    trained_model = state[0]
    return trained_model, losses



# %% [markdown]
# Let's run the first experiment

# %%
config = Config(n_instances=10, n_features=5, n_hidden=2)

model = Model.from_config(config=config)

axis = (0, 2)
probability = jnp.expand_dims(0.9 ** jnp.arange(config.n_instances), axis=axis)
importance = jnp.expand_dims(20 ** -jnp.linspace(0, 1, config.n_instances), axis=axis)

data_generator = DataGenerator(
    n_features=config.n_features,
    feature_probability=probability,
    feature_importance=importance,
    batch_size=1024,
    key=jax.random.key(98923),
)

model_optimized = optimize(model=model, data_generator=data_generator)

# %%
