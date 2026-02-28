# Copyright 2024 The JAX Authors.
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
"""Fixed point primitive."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, TypeVar

import numpy as np

from jax import custom_jvp
from jax._src import ad_util
from jax._src import api_util
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import linear_util as lu
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import pxla
from jax._src.lax import lax
from jax._src.lax.control_flow.loops import while_loop
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.traceback_util import api_boundary
from jax._src.tree_util import tree_flatten, tree_unflatten, FlatTree

T = TypeVar("T")

@api_boundary
def fixed_point(f: Callable[[T, ...], T], x0: T, *args: Any, tol: float = 1e-5, max_iter: int = 100) -> T:
  """Finds a fixed point of f(x, *args) = x using iterative substitution."""
  x0_flat, x_tree = tree_flatten(x0)
  args_flat, args_tree = tree_flatten(args)
  
  def flat_f_internal(cur_x_flat, cur_args_flat):
    cur_x = tree_unflatten(x_tree, cur_x_flat)
    cur_args = tree_unflatten(args_tree, cur_args_flat)
    out = f(cur_x, *cur_args)
    out_flat, _ = tree_flatten(out)
    return out_flat

  res_flat = _fixed_point_flat_wrapper(flat_f_internal, tol, max_iter, len(x0_flat), x0_flat, args_flat)
  
  # Ensure res_flat is a list/tuple for tree_unflatten
  res_flat_list = list(res_flat) if len(x0_flat) > 1 else [res_flat]
  
  return tree_unflatten(x_tree, res_flat_list)

@partial(custom_jvp, nondiff_argnums=(0, 1, 2, 3))
def _fixed_point_flat_wrapper(flat_f, tol, max_iter, num_x, x0_flat, args_flat):
  f_debug = api_util.debug_info("fixed_point", flat_f, (x0_flat, args_flat), {})
  avals_in = [core.get_aval(x) for x in x0_flat] + [core.get_aval(a) for a in args_flat]
  
  def traced_f(*flat_args):
    return flat_f(flat_args[:num_x], flat_args[num_x:])

  f_jaxpr, _, f_consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(traced_f, debug_info=f_debug), avals_in)

  out_flat = fixed_point_p.bind(
      *f_consts, *x0_flat, *args_flat, 
      jaxpr=f_jaxpr, 
      num_consts=len(f_consts),
      num_x=num_x,
      tol=tol, 
      max_iter=max_iter
  )
  if num_x == 1:
    return out_flat[0]
  return out_flat

@_fixed_point_flat_wrapper.defjvp
def _fixed_point_flat_jvp(flat_f, tol, max_iter, num_x, primals, tangents):
  # x0_flat, args_flat = primals
  # dx0_flat, dargs_flat = tangents
  x0_flat, args_flat = primals
  dx0_flat, dargs_flat = tangents
  
  x_star_flat = _fixed_point_flat_wrapper(flat_f, tol, max_iter, num_x, x0_flat, args_flat)
  # Wrap x_star_flat in a list if it's a single value
  x_star_flat_list = [x_star_flat] if num_x == 1 else x_star_flat
  
  def tangent_body_flat(v_flat):
    # JVP of flat_f(x_flat, args_flat) at (x_star_flat, args_flat) with tangents (v_flat, dargs_flat)
    dbg = api_util.debug_info("fixed_point_tangent_body", flat_f, (x_star_flat_list, args_flat), {})
    _, out_tangent = ad.jvp(lu.wrap_init(flat_f, debug_info=dbg)).call_wrapped((x_star_flat_list, args_flat), (v_flat, dargs_flat))
    return out_tangent

  # We use a while_loop for the tangent calculation to avoid nested custom_jvp issues
  def cond_fun(carry):
    i, v, prev_v = carry
    # Use lax.reduce_sum for tracers
    def total_abs_diff(a, b):
      return lax.reduce_sum(lax.abs(lax.sub(a, b)), axes=tuple(range(len(a.shape))))
    
    diff = sum(total_abs_diff(a, b) for a, b in zip(v, prev_v))
    return lax.bitwise_and(i < max_iter, diff > tol)

  def body_fun(carry):
    i, v, _ = carry
    return i + 1, tangent_body_flat(v), v

  init_v = dx0_flat
  # prev_v is just something different from init_v to start the loop
  prev_v = [lax.add(x, lax.full_like(x, 1.0)) for x in init_v]
  
  _, dx_star_flat_list, _ = while_loop(cond_fun, body_fun, (0, init_v, prev_v))
  
  dx_star_flat = dx_star_flat_list[0] if num_x == 1 else dx_star_flat_list
  return x_star_flat, dx_star_flat

fixed_point_p = core.Primitive("fixed_point")
fixed_point_p.multiple_results = True

@fixed_point_p.def_abstract_eval
def _fixed_point_abstract_eval(*avals, jaxpr, num_consts, num_x, **params):
  return avals[num_consts : num_consts + num_x]

def _fixed_point_impl(*args, jaxpr, num_consts, num_x, tol, max_iter):
  breakpoint()
  consts = args[:num_consts]
  x = list(args[num_consts : num_consts + num_x])
  others = args[num_consts + num_x:]
  for _ in range(max_iter):
    next_x = core.eval_jaxpr(jaxpr, consts, *x, *others)
    diffs = [np.sum(np.abs(np.array(a) - np.array(b))) for a, b in zip(x, next_x)]
    x = next_x
    if sum(diffs) < tol:
      break
  return x

fixed_point_p.def_impl(_fixed_point_impl)

def _fixed_point_batching_rule(axis_data, args, dims, jaxpr, num_consts, num_x, tol, max_iter):
  in_batched = [d is not batching.not_mapped for d in dims]
  out_batched = [b for b in in_batched[num_consts : num_consts + num_x]]
  for _ in range(1 + num_x):
    batched_jaxpr, new_out_batched = batching.batch_jaxpr(
        core.ClosedJaxpr(jaxpr, ()), axis_data, in_batched, instantiate=out_batched)
    if new_out_batched == out_batched:
      break
    out_batched = new_out_batched
  new_args = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped else x
              for x, d in zip(args, dims)]
  target_batched = in_batched[:num_consts] + out_batched + in_batched[num_consts + num_x:]
  new_args = [
      batching.broadcast(x, axis_data.size, 0, axis_data.explicit_mesh_axis) if now_bat and not was_bat
      else x
      for x, was_bat, now_bat in zip(new_args, in_batched, target_batched)
  ]
  outs = fixed_point_p.bind(
      *new_args, jaxpr=batched_jaxpr.jaxpr, num_consts=num_consts, num_x=num_x, tol=tol, max_iter=max_iter
  )
  out_dims = [0 if b else batching.not_mapped for b in out_batched]
  return outs, out_dims

batching.fancy_primitive_batchers[fixed_point_p] = _fixed_point_batching_rule

def _fixed_point_lowering(ctx, *args, jaxpr, num_consts, num_x, tol, max_iter):
  f_consts = args[:num_consts]
  x_init = args[num_consts : num_consts + num_x]
  args_const = args[num_consts + num_x:]
  iter_init = mlir.ir_constant(np.int32(0))
  carry_init = [iter_init, *x_init]
  carry_types = [c.type for c in carry_init]
  while_op = hlo.WhileOp(carry_types, mlir.flatten_ir_values(carry_init))
  cond_block = while_op.regions[0].blocks.append(*mlir.flatten_ir_types(carry_types))
  with ir.InsertionPoint(cond_block):
    curr_iter = cond_block.arguments[0]
    max_iter_const = mlir.ir_constant(np.int32(max_iter))
    cond = hlo.CompareOp(curr_iter, max_iter_const, 
                         comparison_direction=hlo.ComparisonDirectionAttr.get("LT")).result
    hlo.return_([cond])
  body_block = while_op.regions[1].blocks.append(*mlir.flatten_ir_types(carry_types))
  with ir.InsertionPoint(body_block):
    curr_iter = body_block.arguments[0]
    curr_x = body_block.arguments[1:]
    name_stack = ctx.name_stack.extend("fixed_point_body")
    out_vals, _ = mlir.jaxpr_subcomp(
        ctx.module_context, jaxpr, name_stack, 
        mlir.TokenSet(), f_consts, *curr_x, *args_const,
        dim_var_values=ctx.dim_var_values, const_lowering=ctx.const_lowering
    )
    next_iter = hlo.AddOp(curr_iter, mlir.ir_constant(np.int32(1))).result
    hlo.return_(mlir.flatten_ir_values([next_iter, *out_vals]))
  results = mlir.unflatten_ir_values_like_types(while_op.results, carry_types)
  return results[1:]

mlir.register_lowering(fixed_point_p, _fixed_point_lowering)
