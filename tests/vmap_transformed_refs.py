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

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax._src import config

config.parse_flags_with_absl()

_cond = lambda fn: lambda *args: jax.lax.cond(jnp.array(True), fn, fn, *args)


class VmapTransformedRefsTest(jtu.JaxTestCase):

  @jtu.parameterized.named_parameters(("vmap", jax.vmap), ("remat", jax.remat), ("cond", _cond), ("jit", jax.jit))
  def test_read_transformed_ref(self, transform):
    """vmap should be able to read from a TransformedRef (ref.at[slice])."""
    @jax.jit
    def f():
      x_ref = jax.new_ref(jnp.arange(10, dtype=jnp.float32))
      x_view = x_ref.at[2:7]  # TransformedRef, shape (5,)
      result = transform(lambda v: v[...])(x_view)
      return result
    expected = jnp.arange(10, dtype=jnp.float32)[2:7]
    self.assertAllClose(f(), expected)

  @jtu.parameterized.named_parameters(("vmap", jax.vmap), ("remat", jax.remat), ("cond", _cond), ("jit", jax.jit))
  def test_write_transformed_ref(self, transform):
    """vmap/remat should be able to write to a TransformedRef (ref.at[slice])."""
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

  @jtu.parameterized.named_parameters(("vmap", jax.vmap), ("remat", jax.remat), ("cond", _cond), ("jit", jax.jit))
  def test_addupdate_transformed_ref(self, transform):
    """vmap/remat should be able to addupdate a TransformedRef."""
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

  # def test_vmap_transformed_ref_closed_over(self):
  #   """vmap over a function that closes over a TransformedRef."""
  #   @jax.jit
  #   def f():
  #     x_ref = jax.new_ref(jnp.zeros(10, dtype=jnp.float32))
  #     x_view = x_ref.at[3:8]
  #     vals = jnp.arange(5, dtype=jnp.float32)
  #     def body(v):
  #       x_view[...] += v * jnp.ones(5)
  #     jax.vmap(body)(vals[None])  # batch dim of size 1
  #     return x_ref[...]
  #   # just check it runs without error for now
  #   f()


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
