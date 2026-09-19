"""Fusible matmul on Mosaic GPU (sm_120: TMA + Ampere-style mma, no WGMMA/TMEM)."""
import functools
import unittest
import jax
from jax import lax
import jax.numpy as jnp
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
from jax.experimental.pallas import mosaic_gpu as plgpu
import numpy as np


def _is_ref(x):
  return isinstance(x, state_types.TransformedRef) or isinstance(getattr(x, "aval", None), state_types.AbstractRef)


def _deref(x):
  return x[...] if _is_ref(x) else x


def _block_shape(spec):
  return tuple(b if b is None else int(getattr(b, "block_size", b)) for b in spec.block_shape)


def _transforms(bshape, dtype):
  """128B swizzle for 2D tiles whose minor dim is a whole number of swizzle atoms, else let MGPU infer."""
  itemsize = jnp.dtype(dtype).itemsize
  if len(bshape) < 2 or None in bshape[-2:] or bshape[-2] % 8 or (bshape[-1] * itemsize) % 128:
    return ()
  return (plgpu.TilingTransform((8, 128 // itemsize)), plgpu.SwizzleTransform(128))


def _fusible_matmul(x, y, z=None, *, bm=64, bk=64, bn=64, stages=2, use_transforms=True):
  (m, k), (_, n) = x.shape, y.shape
  out_dtype = jnp.bfloat16
  z_type = jax.ShapeDtypeStruct((m, n), dtype=out_dtype)
  z = z if z else (lambda v: v)
  grid = (m // bm, n // bn, k // bk)

  x_fn, x_values, x_sp = fuser.get_stateful_input_fusion_values(x)
  y_fn, y_values, y_sp = fuser.get_stateful_input_fusion_values(y)
  z_fn, z_values, z_sp, z_aliases = fuser.get_stateful_output_fusion_values(z, z_type)
  sp_flat, sp_tree = jax.tree.flatten((x_sp, y_sp, z_sp))
  # discharged fusions take the arrays the outer Refs hold, so trace with those types
  avals = lambda vals: jax.tree.map(lambda v: getattr(jax.typeof(v), "inner_aval", jax.typeof(v)), vals)
  sp_handler = lambda i, sp: fuser.make_scalar_prefetch_handler(i) if sp else None

  x_seed = pl.BlockSpec((bm, bk), lambda mi, ni, ki, *_: (mi, ki))
  y_seed = pl.BlockSpec((bk, bn), lambda mi, ni, ki, *_: (ki, ni))
  z_seed = pl.BlockSpec((bm, bn), lambda mi, ni, ki, *_: (mi, ni))
  x_fn, (x_specs,), _ = fuser.pull_block_spec(x_fn, x_seed, scalar_prefetch_handler=sp_handler(0, x_sp), grid_len=3)(
    avals(x_values)
  )
  y_fn, (y_specs,), _ = fuser.pull_block_spec(y_fn, y_seed, scalar_prefetch_handler=sp_handler(1, y_sp), grid_len=3)(
    avals(y_values)
  )
  z_fn, z_specs, _, z_out_type, z_out_spec = fuser.push_pull_block_spec(
    z_fn, z_seed, scalar_prefetch_handler=sp_handler(2, z_sp), grid_len=3
  )(avals(z_values), z_type)
  z_out_leaves, z_out_specs = jax.tree.leaves(z_out_type), jax.tree.leaves(z_out_spec)
  unaliased = [i for i in range(len(z_out_leaves)) if i not in z_aliases]
  sizes = [len(x_values), len(y_values), len(z_values), len(sp_flat)]

  @plgpu.kernel(out_type=tuple(z_out_leaves[i] for i in unaliased), grid=grid[:2], grid_names=("m", "n"))
  def kernel(*refs):
    x_refs, y_refs, z_refs, sp_refs, out_refs = (
      refs[sum(sizes[:i]) : sum(sizes[: i + 1])] if i < 4 else refs[sum(sizes) :] for i in range(5)
    )
    mi, ni = lax.axis_index("m"), lax.axis_index("n")
    sp = sp_tree.unflatten([r[...] for r in sp_refs])
    # blocked values go through the TMA pipeline, unblocked ones are handed to the fusion as GMEM refs
    in_refs, in_specs = (*x_refs, *y_refs), (*x_specs, *y_specs)
    piped = [(r, s) for r, s in zip(in_refs, in_specs) if s is not pl.no_block_spec]
    slc = lambda spec: tuple(
      i if b is None else pl.ds(i * b, b) for i, b in zip(spec.index_map(mi, ni, 0, *sp), _block_shape(spec))
    )
    pipe_spec = lambda r, s, k0: plgpu.BlockSpec(
      _block_shape(s),
      lambda kl: s.index_map(mi, ni, k0 + kl, *sp),
      transforms=_transforms(_block_shape(s), r.dtype) if use_transforms else (),
    )

    def body(idxs, *smem_and_acc, active, k0):
      (kl,), (*smem, acc) = idxs, smem_and_acc
      smem, ki = iter(smem), k0 + kl
      piped_tiles = iter([next(smem) if a else None for a in active])
      tiles = [next(piped_tiles) if s is not pl.no_block_spec else r for r, s in zip(in_refs, in_specs)]
      x_val = x_fn((mi, ni, ki), sp, tuple(tiles[: len(x_refs)]))
      y_val = y_fn((mi, ni, ki), sp, tuple(tiles[len(x_refs) :]))
      x_val, y_val = (jax.tree.leaves(v, is_leaf=_is_ref)[0] for v in (x_val, y_val))
      return plgpu.mma(acc, _deref(x_val), _deref(y_val))

    # Partial-block concat children carry BlockSelect entries: a child is only needed on grid steps where its
    # group's select_fn picks it. Groups that vary along k split the pipeline into segments; groups that vary
    # with the CTA (m/n) become a lax.switch with one specialized pipeline per child.
    selects = {sel.group: sel for _, s in piped for sel in getattr(s, "select", ())}
    choose = lambda group, ki: selects[group].select_fn(mi, ni, ki, *sp)
    is_active = lambda s, chosen: all(chosen[sel.group] == sel.child for sel in getattr(s, "select", ()))

    def run_segments(acc, fixed):
      segments = []
      for ki in range(grid[2]):
        chosen = {g: fixed.get(g, choose(g, ki)) for g in selects}
        assert not any(isinstance(c, jax.core.Tracer) for c in chosen.values()), "dynamic selection along k"
        if segments and segments[-1][2] == chosen:
          segments[-1][1] = ki + 1
        else:
          segments.append([ki, ki + 1, chosen])
      for k0, k1, chosen in segments:
        active = [is_active(s, chosen) for _, s in piped]
        acc = plgpu.emit_pipeline(
          functools.partial(body, active=active, k0=k0),
          grid=(k1 - k0,),
          in_specs=[pipe_spec(r, s, k0) for (r, s), a in zip(piped, active) if a],
          max_concurrent_steps=stages,
          init_carry=acc,
        )(*[r for (r, _), a in zip(piped, active) if a])
      return acc

    def resolve(acc, groups, fixed):
      if not groups:
        return run_segments(acc, fixed)
      g, *rest = groups
      branches = [functools.partial(resolve, groups=rest, fixed={**fixed, g: j}) for j in range(selects[g].num_children)]
      return lax.switch(choose(g, 0), branches, acc)

    dynamic = [g for g in selects if isinstance(choose(g, 0), jax.core.Tracer)]
    acc = resolve(jnp.zeros((bm, bn), jnp.float32), dynamic, {})

    # blocked output-fusion values are staged GMEM -> SMEM once, the fusion sees them as SMEM refs
    z_piped = [(r, s) for r, s in zip(z_refs, z_specs) if s is not pl.no_block_spec]

    def epilogue(*z_smem_and_barrier):
      *z_smem, barrier = z_smem_and_barrier
      for i, ((r, s), smem) in enumerate(zip(z_piped, z_smem)):
        plgpu.copy_gmem_to_smem(r.at[slc(s)], smem, barrier.at[i])
      for i in range(len(z_piped)):
        plgpu.barrier_wait(barrier.at[i])
      smem = iter(z_smem)
      z_vals = tuple(next(smem) if s is not pl.no_block_spec else r for r, s in zip(z_refs, z_specs))
      outs = iter(jax.tree.leaves(z_fn((mi, ni, 0), sp, z_vals, acc.astype(out_dtype)), is_leaf=_is_ref))
      for i, spec in enumerate(z_out_specs):
        target = z_refs[z_aliases[i]] if i in z_aliases else out_refs[unaliased.index(i)]
        # a MultiBlockSpec output gets one tile per child, each stored at its own offset
        for s in getattr(spec, "specs", (spec,)):
          target[slc(s)] = _deref(next(outs)).astype(target.dtype)

    pl.run_scoped(
      epilogue,
      *[plgpu.SMEM(_block_shape(s), r.dtype, transforms=_transforms(_block_shape(s), r.dtype)) for r, s in z_piped],
      plgpu.Barrier(num_barriers=max(len(z_piped), 1)),
    )

  out = kernel(*x_values, *y_values, *z_values, *sp_flat)
  return out[0] if len(unaliased) == 1 else out


def fusible_matmul(x, y, **kw):
  return fuser.fusible(lambda *args, **kwargs: _fusible_matmul(*args, **kw, **kwargs))(x, y)


class MgpuFusibleMatmulTest(unittest.TestCase):
  def setUp(self):
    if jax.devices()[0].platform != "gpu":
      self.skipTest("Mosaic GPU only")
    keys = iter(jax.random.split(jax.random.key(0), 16))
    self.rand = lambda shape: jax.random.uniform(next(keys), shape, dtype=jnp.bfloat16)
    self.x, self.w = self.rand((128, 64)), self.rand((64, 128))

  def check(self, fn, ref, *args):
    out, expected = jax.jit(fuser.fuse(fn))(*args), ref(*args)
    np.testing.assert_allclose(out.astype(jnp.float32), expected.astype(jnp.float32), atol=1e-1, rtol=1e-2)

  def test_identity(self):
    for use_transforms in (True, False):
      self.check(lambda x, w: fusible_matmul(x, w, use_transforms=use_transforms), lambda x, w: x @ w, self.x, self.w)

  def test_fused_inputs(self):
    self.check(lambda x, w: fusible_matmul(x * 2.0, w * 0.5), lambda x, w: (x * 2.0) @ (w * 0.5), self.x, self.w)

  def test_fused_elementwise_outputs(self):
    for f in (jax.nn.relu, lambda z: jax.nn.gelu(z) * 0.75, lambda z: z * jax.nn.sigmoid(z),
              lambda z: (z * 1.5 + 2.0) ** 2, lambda z: jnp.clip(z, 0.2, 0.8)):
      self.check(lambda x, w: f(fusible_matmul(x, w)), lambda x, w: f(x @ w), self.x, self.w)

  def test_fused_output_bias(self):
    b = self.rand((128,))
    self.check(lambda x, w, b: fusible_matmul(x, w) + b[None, :], lambda x, w, b: x @ w + b[None, :], self.x, self.w, b)

  def test_transposed_operands(self):
    xt, wt = self.x.T, self.w.T
    self.check(lambda xt, w: fusible_matmul(xt.T, w), lambda xt, w: xt.T @ w, xt, self.w)
    self.check(lambda x, wt: fusible_matmul(x, wt.T), lambda x, wt: x @ wt.T, self.x, wt)
    self.check(lambda xt, wt: fusible_matmul(xt.T, wt.T), lambda xt, wt: xt.T @ wt.T, xt, wt)

  def test_fused_input_concat_full_dim(self):
    x1, x2, w1, w2 = self.rand((128, 32)), self.rand((128, 32)), self.rand((64, 64)), self.rand((64, 64))
    self.check(
      lambda x1, x2, w1, w2: fusible_matmul(jnp.concatenate([x1, x2], 1), jnp.concatenate([w1, w2], 1), bn=128),
      lambda x1, x2, w1, w2: jnp.concatenate([x1, x2], 1) @ jnp.concatenate([w1, w2], 1), x1, x2, w1, w2,
    )
    xa, xb = self.rand((64, 64)), self.rand((64, 64))
    self.check(
      lambda xa, xb, w: fusible_matmul(jnp.concatenate([xa, xb], 0), w, bm=128),
      lambda xa, xb, w: jnp.concatenate([xa, xb], 0) @ w, xa, xb, self.w,
    )

  def test_fused_input_concat_partial_block(self):
    x1, x2, w1, w2 = self.rand((128, 128)), self.rand((128, 128)), self.rand((64, 128)), self.rand((64, 128))
    xa, xb, w_tall = self.rand((128, 64)), self.rand((128, 64)), self.rand((256, 128))
    cat = jnp.concatenate
    cases = {  # name: (fn(matmul, *args), args)
      "K: segmented pipeline": (lambda mm, x1, x2, w: mm(cat([x1, x2], 1), w), (x1, x2, w_tall)),
      "K with child elementwise": (lambda mm, x1, x2, w: mm(cat([x1 * 2.0, x2], 1), w), (x1, x2, w_tall)),
      "M: switch on lhs": (lambda mm, xa, xb, w: mm(cat([xa, xb], 0), w), (xa, xb, self.w)),
      "N: switch on rhs": (lambda mm, x, w1, w2: mm(x, cat([w1, w2], 1)), (self.x, w1, w2)),
      "M and N": (lambda mm, xa, xb, w1, w2: mm(cat([xa, xb], 0), cat([w1, w2], 1)), (xa, xb, w1, w2)),
    }
    for name, (fn, args) in cases.items():
      with self.subTest(name):
        self.check(functools.partial(fn, fusible_matmul), functools.partial(fn, jnp.matmul), *args)

  def test_fused_output_concat_partial_block(self):
    for axis in (1, 0):
      self.check(
        lambda x, w: jnp.concatenate([jax.nn.relu(z := fusible_matmul(x, w)), z * 2.0], axis),
        lambda x, w: jnp.concatenate([jax.nn.relu(x @ w), (x @ w) * 2.0], axis), self.x, self.w,
      )

  def test_fused_input_split(self):
    x_wide = self.rand((128, 128))
    self.check(
      lambda xw, w: fusible_matmul(lax.split(xw, (64, 64), axis=1)[1], w), lambda xw, w: xw[:, 64:] @ w, x_wide, self.w
    )

  def test_fused_output_concat(self):
    self.check(
      lambda x, w: jnp.concatenate([fusible_matmul(x, w, bn=128)] * 2, 1), lambda x, w: jnp.concatenate([x @ w] * 2, 1),
      self.x, self.w,
    )

  def test_fused_output_split(self):
    for axis, kw in ((1, dict(bn=128)), (0, dict(bm=128))):
      split = lambda z: lax.split(z, (64, 64), axis=axis)
      self.check(
        lambda x, w: (lambda z1, z2: z1 + z2)(*split(fusible_matmul(x, w, **kw))),
        lambda x, w: (lambda z1, z2: z1 + z2)(*split(x @ w)), self.x, self.w,
      )

  def test_multi_tile_grid(self):
    x, w = self.rand((256, 128)), self.rand((128, 256))
    self.check(lambda x, w: jax.nn.relu(fusible_matmul(x * 2.0, w)), lambda x, w: jax.nn.relu((x * 2.0) @ w), x, w)

  def test_stateful_input_and_output(self):
    offset, size = 64, 64

    @jax.jit
    def run(x, w):
      x_ref, out_ref = jax.new_ref(x), jax.new_ref(jax.lax.empty((128, 128), jnp.bfloat16))
      off = jnp.array(offset)

      @fuser.fuse
      def matmul():
        out_ref[pl.ds(off, size), :] = fusible_matmul(x_ref[pl.ds(off, size), :] * 2.0, w)

      matmul()
      return jax.freeze(out_ref)

    res = run(self.x, self.w)[offset : offset + size]
    expected = (self.x[offset : offset + size] * 2.0) @ self.w
    np.testing.assert_allclose(res.astype(jnp.float32), expected.astype(jnp.float32), atol=1e-1, rtol=1e-2)


if __name__ == "__main__":
  unittest.main()
