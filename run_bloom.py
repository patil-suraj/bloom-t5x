import time
import warnings

import jax
import numpy as np
import jax.numpy as jnp
from flax.core.frozen_dict import freeze
from jax.experimental import PartitionSpec as P
from jax.experimental import multihost_utils
from t5x.partitioning import PjitPartitioner
from t5x.train_state import InferenceState

from modeling_flax_bloom import FlaxBloomForCausalLM, BloomConfig
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ResourceWarning)

if jax.process_index() == 0:
    warnings.filterwarnings("default")

# print but only on the first node
def head_print(*args, **kwargs):
    if jax.process_index() == 0:
        print(*args, **kwargs)

start = time.time()
jax.devices()
head_print(f"jax devices: {jax.device_count()}")
head_print(f"jax runtime initialized in {time.time() - start:.06}s")

head_print("Loading model")
ckpt = "bigscience/bloom"
config = BloomConfig.from_pretrained(ckpt)
model = FlaxBloomForCausalLM(config, _do_init=False, dtype=jnp.bfloat16, params_dtype=jnp.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-350m", use_fast=False)
head_print("Loading complete")


# 1D parameter partitioning with 2D activation partitioning
logical_axis_rules = [
    ('batch', 'data'),
    ('mlp', 'model'),
    ('heads', 'model'),
    ('vocab', 'model'),
    # shard remaining activations; weight matrices already have axes mapped to 'model'
    ('embed', 'model'),
    ('kv', None),
    ('joined_kv', None),
    ('relpos_buckets', None),
    ('abspos_buckets', None),
    ('length', None),
    ('layers', None),
    ('stack', None),
    ('mlp_activations', None),
]

# 2D parameter and activation partitioning
logical_axis_rules_full = [
    ('batch', 'data'),
    ('mlp', 'model'),
    ('heads', 'model'),
    ('vocab', 'model'),
    # shard both activations and weight matrices on the remaining available axis
    ('embed', 'model'),
    ('embed', 'data'),
    ('kv', None),
    ('joined_kv', None),
    ('relpos_buckets', None),
    ('abspos_buckets', None),
    ('length', None),
    ('layers', None),
    ('stack', None),
    ('mlp_activations', None),
]


# TODO: Add this in model init
def init_fn():
    input_shape = (1,1)
    input_ids = jnp.zeros(input_shape, dtype="i4")
    attention_mask = jnp.ones_like(input_ids)
    rng = jax.random.PRNGKey(0)
    return model.module.init(rng, input_ids, attention_mask, return_dict=False)


param_axes = jax.eval_shape(init_fn)["params_axes"] # Axis names metadata
# create InferenceState, since the partitioner expects it. 
state = InferenceState(
    step=jnp.array(0),
    params=freeze(model.params_shape_tree),
    params_axes=freeze(param_axes),
    flax_mutables=None,
    flax_mutables_axes=param_axes,
)

# (2, 4, 4, 1)
num_mp_partitions = 8
model_parallel_submesh = (2, 4, 1, 1)
partitioner = PjitPartitioner(
    model_parallel_submesh=model_parallel_submesh,
    logical_axis_rules=logical_axis_rules_full
)
data_layout = partitioner.get_data_layout(4)
mesh_axes = partitioner.get_mesh_axes(state)
params_spec = mesh_axes.params

def init_params():
    input_shape = (data_layout.batch_size, 16)
    input_ids = jnp.zeros(input_shape, dtype="i4")
    attention_mask = jnp.ones_like(input_ids)
    rng = jax.random.PRNGKey(0)
    return model.module.init(rng, input_ids, attention_mask, return_dict=False)["params"]

p_init_params = partitioner.partition(init_params, None, params_spec)
multihost_utils.sync_global_devices("spec created")

head_print("initilizing params")
params = p_init_params()
# Block until complete on all hosts.
multihost_utils.sync_global_devices("sharded over pod")
head_print(f"Initialized in {time.time() - start:.06}s")
