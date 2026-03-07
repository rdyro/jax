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

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax._src import config
from jax.sharding import AxisType, PartitionSpec as P

config.parse_flags_with_absl()

_cond = lambda fn: lambda *args: jax.lax.cond(jnp.array(True), fn, fn, *args)

def shard_map(fn, out_specs):
  mesh = jax.make_mesh((jax.device_count(),), ("x",), axis_types=(AxisType.Explicit,))
  return jax.shard_map(fn, out_specs=out_specs, mesh=mesh)


class TransformedRefsTest(jtu.JaxTestCase):

  @parameterized.named_parameters(("jit", "jit"), ("remat", "remat"), ("shard_map", "shard_map"))
  def test_transformed_ref_in(self, transform):

    transform = {"jit": jax.jit, "remat": jax.remat,
                 "shard_map": partial(shard_map, out_specs=None)}[transform]

    @jax.jit
    def f():
      x_ref = jax.new_ref(jnp.zeros(7))
      x_ref_view = x_ref.at[1:6]
      @transform
      def inner(x_ref_view):
        x_ref_view[...] += 17 * jnp.ones((5,))
      inner(x_ref_view)
      return x_ref[...]
    expected = jnp.zeros(7).at[1:6].add(17.0)
    self.assertAllClose(f(), expected)

  def test_transformed_ref_in_cond(self):
    @jax.jit
    def f(pred):
      x_ref = jax.new_ref(jnp.zeros(7))
      x_ref_view = x_ref.at[1:6]
      def true_fn(x_ref_view):
        x_ref_view[...] += 17 * jnp.ones((5,))
      def false_fn(x_ref_view):
        x_ref_view[...] += 5 * jnp.ones((5,))
      jax.lax.cond(pred, true_fn, false_fn, x_ref_view)
      return x_ref[...]

    expected_true = jnp.zeros(7).at[1:6].add(17.0)
    expected_false = jnp.zeros(7).at[1:6].add(5.0)

    self.assertAllClose(f(True), expected_true)
    self.assertAllClose(f(False), expected_false)

  @parameterized.named_parameters(
      ("remat", jax.remat), ("cond", _cond), ("jit", jax.jit),
      ("shard_map", partial(shard_map, out_specs=P())))
  def test_read_transformed_ref(self, transform):
    @jax.jit
    def f():
      x_ref = jax.new_ref(jnp.arange(10, dtype=jnp.float32))
      x_view = x_ref.at[2:7]  # TransformedRef, shape (5,)
      result = transform(lambda v: v[...])(x_view)
      return result
    expected = jnp.arange(10, dtype=jnp.float32)[2:7]
    self.assertAllClose(f(), expected)

  @parameterized.named_parameters(("false", False), ("true", True))
  def test_temp(self, pure_ref):
    """remat should be able to write to a TransformedRef (ref.at[slice])."""
    x_ref = jax.new_ref(jnp.zeros(10, dtype=jnp.float32))
    @jax.jit
    def f(x_ref):
      x_view = x_ref.at[2:7]
      def body(v):
        if pure_ref:
          v[2:7] = jnp.ones_like(v[2:7])
        else:
          v[...] = jnp.ones_like(v)
      if pure_ref:
        _cond(body)(x_ref)
      else:
        _cond(body)(x_view)
      return x_ref[...]
    expected = jnp.zeros(10).at[2:7].set(1.0)
    self.assertAllClose(f(x_ref), expected)

  @parameterized.named_parameters(
      ("remat", jax.remat), ("cond", _cond), ("jit", jax.jit),
      ("shard_map", partial(shard_map, out_specs=None))
  )
  def test_write_transformed_ref(self, transform):
    """remat should be able to write to a TransformedRef (ref.at[slice])."""
    @jax.jit
    def f():
      x_ref = jax.new_ref(jnp.zeros(10, dtype=jnp.float32))
      x_view = x_ref.at[2:7]
      def body(v):
        v[...] = jnp.ones_like(v)
      transform(body)(x_view)
      return x_ref[...]
    expected = jnp.zeros(10).at[2:7].set(1.0)
    self.assertAllClose(f(), expected)

  @parameterized.named_parameters(
      ("remat", jax.remat), ("cond", _cond), ("jit", jax.jit),
      ("shard_map", partial(shard_map, out_specs=None))
  )
  def test_addupdate_transformed_ref(self, transform):
    """remat should be able to addupdate a TransformedRef."""
    @jax.jit
    def f():
      x_ref = jax.new_ref(jnp.ones(10, dtype=jnp.float32))
      x_view = x_ref.at[2:7]
      def body(v):
        v[...] += 1.0
      transform(body)(x_view)
      return x_ref[...]
    expected = jnp.ones(10).at[2:7].add(1.0)
    self.assertAllClose(f(), expected)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())