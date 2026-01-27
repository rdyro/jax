# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Internal Tokamax Megablox TPU benchmarking and tuning."""

# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import

from absl import app
import jax
from jax import numpy as jnp
from jax import random
from jax.experimental import xla_metadata
import numpy as np
import tune_jax

from jax._src.lax.pallas_lowering import _pallas_ragged_dot_general_impl
from jax._src.lax.lax import ragged_dot_general_p, mlir

mlir.register_lowering(
  ragged_dot_general_p,
  mlir.lower_fun(_pallas_ragged_dot_general_impl, multiple_results=False),
  platform="cpu",
)

jax.config.update("jax_traceback_filtering", "off")


def generate_group_sizes(key, *, target_m: int, g: int) -> jax.Array:
  """Generate group sizes for a given target m."""
  assert target_m >= 0
  if target_m == 0:
    return jnp.zeros(g, dtype=jnp.int32)
  gs = jnp.round(
      target_m * jax.nn.softmax(3e-1 * random.normal(key, (g,)))
  ).astype(jnp.int32)
  while jnp.sum(gs) != target_m:
    idx = jnp.argmax(gs)
    gs = gs.at[idx].set(jnp.maximum(target_m - jnp.sum(gs) + gs[idx], 0))
  return gs


def generate_gmm_inputs(
    key,
    *,
    m: int,
    n: int,
    k: int,
    g: int,
    random_values: bool = True,
    dtype: jnp.dtype = jnp.bfloat16,
    jax_random: bool = False,
):
  """Generate group sizes for a given target m."""

  keys = iter(random.split(key, 16))

  def randn_init(key, shape, dtype):
    if jax_random:
      return random.normal(key, shape, dtype=dtype)
    seed = sum(int(k) << (32 * i) for i, k in enumerate(random.key_data(key)))
    return jnp.array(np.random.default_rng(seed).normal(size=shape), dtype)

  def init_fn(shape):
    if random_values:
      return randn_init(next(keys), shape, dtype) / (shape[-1] ** 0.5)
    else:
      return jnp.ones(shape, dtype=dtype) / (shape[-1] ** 0.5)

  lhs, rhs, dout = init_fn((m, k)), init_fn((g, k, n)), init_fn((m, n))
  # scale some of the inputs to test scales are mapped properly in quantization
  lhs = lhs.at[::3, : lhs.shape[1] // 2].set(lhs[::3, : lhs.shape[1] // 2] * 3)
  rhs = rhs.at[:, :, : rhs.shape[2] // 2].set(
      rhs[:, :, : rhs.shape[2] // 2] * 10
  )
  return lhs, rhs, dout


# ragged dot variants ##########################################################
def lax_gmm_fwd(lhs, rhs, group_sizes, tile_m, tile_k, tile_n, **kw):
  del kw
  dims = jax.lax.RaggedDotDimensionNumbers(
      (((1,), (1,)), ((), ())),  # (contracting, batch)
      (0,),  # (lhs ragged dims)
      (0,),  # (rhs group dim)
  )
  with xla_metadata.set_xla_metadata(
      ragged_dot_tiling=f"{tile_m},{tile_k},{tile_n}"
  ):
    return jax.lax.ragged_dot_general(lhs, rhs, group_sizes, dims)


def lax_dlhs_bwd(out, rhs, group_sizes, tile_m, tile_k, tile_n, **kw):
  del kw
  dims = jax.lax.RaggedDotDimensionNumbers(
      (((1,), (1,)), ((), ())),  # (contracting, batch)
      (0,),  # (lhs ragged dims)
      (0,),  # (rhs group dim)
  )
  with xla_metadata.set_xla_metadata(
      ragged_dot_tiling=f"{tile_m},{tile_k},{tile_n}"
  ):
    return jax.lax.ragged_dot_general(out, rhs.mT, group_sizes, dims)


def lax_drhs_bwd(lhs, out, group_sizes, tile_m, tile_k, tile_n, **kw):
  del kw
  dims = jax.lax.RaggedDotDimensionNumbers(
      (((0,), (0,)), ((), ())),  # (contracting, batch)
      (0,),  # (lhs ragged dims)
      (),  # (rhs group dim)
  )
  with xla_metadata.set_xla_metadata(
      ragged_dot_tiling=f"{tile_m},{tile_k},{tile_n}"
  ):
    return jax.lax.ragged_dot_general(lhs, out, group_sizes, dims)


# ragged dot variants ##########################################################

LOG = print

def test_lowering(
    *,
    m: int,
    k: int,
    n: int,
    g: int,
    tune_modes: str,
    tune_samples: int,
):
  keys = iter(random.split(random.key(0), 1024))
  dtype = jnp.bfloat16
  gs = generate_group_sizes(next(keys), target_m=m, g=g)
  opts = dict(jax_random=True, random_values=False)
  lhs, rhs, dout = generate_gmm_inputs( next(keys), m=m, n=n, k=k, g=g, dtype=dtype, **opts)

  opts = dict(tile_m=128, tile_k=128, tile_n=128)
  y = lax_gmm_fwd(lhs, rhs, group_sizes=gs, **opts)
  dlhs = lax_dlhs_bwd(dout, rhs, group_sizes=gs, **opts)
  drhs = lax_drhs_bwd(lhs, dout, group_sizes=gs, **opts)
  
  return


def run_benchmark(
    *,
    m: int,
    k: int,
    n: int,
    g: int,
    tune_modes: str,
    tune_samples: int,
):
  tune_jax.tune_logger.setLevel("INFO")
  tune_jax.CONFIG.allow_fallback_timing = False  # do not allow timing in Python

  modes_to_tune = tune_modes.split(",")
  if "all" in modes_to_tune:
    modes_to_tune = ["fwd", "dlhs", "drhs"]
  keys = iter(random.split(random.key(0), 1024))

  sample_num = tune_samples
  dtype = jnp.bfloat16

  gs = generate_group_sizes(next(keys), target_m=m, g=g)
  opts = dict(jax_random=True, random_values=False)
  lhs, rhs, dout = generate_gmm_inputs( next(keys), m=m, n=n, k=k, g=g, dtype=dtype, **opts)

  # `_ceil` all hyperparameter tiles to be a multiple of 128. This does not make
  # sense in general, but for ragged dot it unlocks full shape tiling when a
  # dimension is not a multiple of 128 in JAX, but is padded in reality.
  _ceil = lambda x, div=128: div * ((x + div - 1) // div)  # ceiling div
  divs = set([k, k // 2, k // 4]) | set([n, n // 2, n // 4])
  divs = set(map(_ceil, divs))
  hyperparams = {
      "tile_m": list(set(map(_ceil, [128, 256, 512, 1024, 2048, 4096]))),
      "tile_k": list(set(map(_ceil, [256, 512, 1024, 2048])) | divs | set([k])),
      "tile_n": list(
          set(map(_ceil, [256, 512, 1024, 2048, n])) | divs | set([n])
      ),
      "input_buffer_count": [2, 3],
  }

  # forward pass ###############################################################
  LOG(f"FWD PASS --- {(m, k, n)=} ----------------------------")
  if "fwd" in modes_to_tune:
    fn_fwd = tune_jax.tune(
        gmm_fwd,
        hyperparams=hyperparams,
        sample_num=sample_num,
        event_filter_regex="gmm",
    )
    _ = fn_fwd(lhs, rhs, gs)
    LOG("Fwd pass results")
    LOG(tune_jax.tabulate(fn_fwd))

    lax_fn_fwd = tune_jax.tune(
        lax_gmm_fwd,
        hyperparams=hyperparams,
        sample_num=sample_num,
        event_filter_regex="ragged",
    )
    _ = lax_fn_fwd(lhs, rhs, gs)
    LOG("LAX Fwd pass results")
    LOG(tune_jax.tabulate(lax_fn_fwd))
  # forward pass ###############################################################

  # dlhs #######################################################################
  LOG("DLHS ----------------------------------------------------------------")
  if "dlhs" in modes_to_tune:
    dlhs_fn = tune_jax.tune(
        dlhs_bwd,
        hyperparams=hyperparams,
        sample_num=sample_num,
        event_filter_regex="gmm",
    )
    _ = dlhs_fn(dout, rhs, gs)
    LOG("DLHS pass results")
    LOG(tune_jax.tabulate(dlhs_fn))

    lax_dlhs_fn = tune_jax.tune(
        lax_dlhs_bwd,
        hyperparams=hyperparams,
        sample_num=sample_num,
        event_filter_regex="ragged",
    )
    _ = lax_dlhs_fn(dout, rhs, gs)
    LOG("LAX DLHS pass results")
    LOG(tune_jax.tabulate(lax_dlhs_fn))
  # dlhs #######################################################################

  # drhs #######################################################################
  LOG("DRHS ----------------------------------------------------------------")
  if "drhs" in modes_to_tune:
    drhs_fn = tune_jax.tune(
        drhs_bwd,
        hyperparams=hyperparams,
        sample_num=sample_num,
        event_filter_regex="tgmm",
    )
    _ = drhs_fn(lhs, dout, gs)
    LOG("DRHS pass results")
    LOG(tune_jax.tabulate(drhs_fn))

    lax_drhs_fn = tune_jax.tune(
        lax_drhs_bwd,
        hyperparams=hyperparams,
        sample_num=sample_num,
        event_filter_regex="ragged",
    )
    _ = lax_drhs_fn(lhs, dout, gs)
    LOG("LAX DRHS pass results")
    LOG(tune_jax.tabulate(lax_drhs_fn))
  # drhs #######################################################################

  # pytype: disable=name-error
  with jax.profiler.trace("/tmp/profiles"):
    if "fwd" in modes_to_tune:
      for _ in range(2):
        jax.block_until_ready(fn_fwd(lhs, rhs, gs))
      for _ in range(3):
        jax.block_until_ready(lax_fn_fwd(lhs, rhs, gs))
    if "dlhs" in modes_to_tune:
      for _ in range(3):
        jax.block_until_ready(dlhs_fn(dout, rhs, gs))
      for _ in range(3):
        jax.block_until_ready(lax_dlhs_fn(dout, rhs, gs))
    if "drhs" in modes_to_tune:
      for _ in range(3):
        jax.block_until_ready(drhs_fn(lhs, dout, gs))
      for _ in range(3):
        jax.block_until_ready(lax_drhs_fn(lhs, dout, gs))

  # pytype: disable=name-error
  # pytype: disable=attribute-error
  print("-------------------------  Results -------------------------")
  print(f"For m={m}, k={k}, n={n}, g={g}")
  if "fwd" in modes_to_tune:
    print(f"FWD: {fn_fwd.optimal_hyperparams=}")
  if "dlhs" in modes_to_tune:
    print(f"DLHS: {dlhs_fn.optimal_hyperparams=}")
  if "drhs" in modes_to_tune:
    print(f"DRHS: {drhs_fn.optimal_hyperparams=}")
  print("------------------------------------------------------------")
  # pytype: enable=attribute-error
  # pytype: enable=name-error


def main(argv):
  del argv
  print(f"Hello from the program, seing {jax.devices()=}")

  m, k, n, g = 4096, 2048, 1024, 32
  tune_modes = "all"
  tune_samples = 8

  opts = dict(tune_modes=tune_modes, tune_samples=tune_samples)
  # run_benchmark(m=m, k=k, n=n, g=g, **opts)
  test_lowering(m=m, k=k, n=n, g=g, **opts)


if __name__ == "__main__":
  app.run(main)
