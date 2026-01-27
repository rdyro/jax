# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import math

from jax._src import ad_util
from jax._src import api
from jax._src import api_util
from jax._src import array
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import literals
from jax._src import linear_util as lu
from jax._src import pjit
from jax._src import pretty_printer as pp
from jax._src import source_info_util
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.typing import Array, ArrayLike, DimSize, DuckTypedArray, DType, DTypeLike, Shape

from jax._src.lax.lax import (
  _ragged_dot_mode_and_dim, _ragged_dot_prefix_dims, RaggedDotMode, PrecisionLike, dot_general,
  RaggedDotDimensionNumbers, remaining, broadcast
)

import numpy as np
import jax.numpy as jnp

def gmm(lhs, rhs, group_sizes, *, trans_rhs: bool, **kw):
  return jnp.ones((lhs.shape[0], rhs.shape[-1]), lhs.dtype)

def tgmm(lhs, dout, group_sizes, **kw):
  return jnp.ones((group_sizes.shape[-1], lhs.shape[1], dout.shape[-1]), lhs.dtype)


def _hyperparam_selection_rule(dtype: DType):
  smem_size = 100 * 1024
  ideal_operand_size = smem_size / dtype.itemsize / 4
  tile_m = 2 ** round(math.ceil(math.log2(math.sqrt(ideal_operand_size))))
  tile_k, tile_n = tile_m // 2, tile_m
  return dict(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)
  
def compose_vmap(fn, times):
  for _ in range(times):
    fn = api.vmap(fn)
  return fn

def _pallas_ragged_dot_general_impl(
    lhs: Array,
    rhs: Array,
    group_sizes: Array,
    ragged_dot_dimension_numbers: RaggedDotDimensionNumbers,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    group_offset: Array | None = None,
) -> Array:
  mode, lhs_ragged_dim = _ragged_dot_mode_and_dim(
      lhs.ndim, ragged_dot_dimension_numbers
  )
  (l_contract, r_contract), (l_batch, r_batch) = (
      ragged_dot_dimension_numbers.dot_dimension_numbers
  )
  _dot_general = partial(
      dot_general,
      precision=precision,
      preferred_element_type=preferred_element_type,
  )
  l_noncontract = tuple(remaining(range(lhs.ndim), l_contract, l_batch))

  if group_sizes.ndim == 1:
    group_sizes = broadcast(group_sizes, [lhs.shape[i] for i in l_batch])

  assert len(l_batch) == len(r_batch)
  assert len(l_contract) == 1 and len(r_contract) == 1
  
  compute_dtype = np.promote_types(lhs.dtype, rhs.dtype)
  opts = dict(out_dtype=preferred_element_type, **_hyperparam_selection_rule(compute_dtype))
  match mode:
    case RaggedDotMode.RAGGED_NONCONTRACTING:
      rhs_group_dims = ragged_dot_dimension_numbers.rhs_group_dimensions
      assert l_noncontract == (lhs_ragged_dim,)
      assert len(l_batch) == len(r_batch)

      r_noncontract = tuple(remaining(range(rhs.ndim), r_contract, r_batch, rhs_group_dims))
      assert len(r_noncontract) == 1

      lhs_perm = l_batch + l_noncontract + l_contract
      if r_contract[0] < r_noncontract[0]:
        trans_rhs = False
        rhs_perm = r_batch + rhs_group_dims + r_contract + r_noncontract
      else:
        trans_rhs = True
        rhs_perm = r_batch + rhs_group_dims + r_noncontract + r_contract

      lhs, rhs = lhs.transpose(lhs_perm), rhs.transpose(rhs_perm)
      batch_axes = tuple(range(len(l_batch)))

      fn = compose_vmap(partial(gmm, trans_rhs=trans_rhs, **opts), len(batch_axes))
      out = fn(lhs, rhs, group_sizes)

      # out: Array = api.vmap(
      #   partial(gmm, trans_rhs=trans_rhs, **opts),
      #   in_axes=3 * (batch_axes,), out_axes=batch_axes
      # )(lhs, rhs, group_sizes)
      
      out = out.transpose(np.argsort(lhs_perm))
      return out
    case RaggedDotMode.RAGGED_CONTRACTING:
      r_noncontract = tuple(remaining(range(rhs.ndim), r_contract, r_batch))
      lhs_perm = l_batch + l_contract + l_noncontract
      rhs_perm = r_batch + r_contract + r_noncontract

      lhs, rhs = lhs.transpose(lhs_perm), rhs.transpose(rhs_perm)
      batch_axes = tuple(range(len(l_batch)))

      # out: Array  = api.vmap(
      #   partial(tgmm, **opts),
      #   in_axes=3 * (batch_axes,), out_axes=batch_axes
      # )(lhs, rhs, group_sizes)  # [b..., g, k..., n...]
      out = compose_vmap(partial(tgmm, **opts), len(l_batch))(lhs, rhs, group_sizes)
      out_perm = np.argsort(l_batch + (lhs_ragged_dim,) + l_noncontract + r_noncontract)
      out = out.transpose(out_perm)
      return out
    case RaggedDotMode.RAGGED_BATCH:
      return _dot_general(
          lhs,
          rhs,
          dimension_numbers=ragged_dot_dimension_numbers.dot_dimension_numbers,
      )  # pytype: disable=bad-return-type