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
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax._src import config

config.parse_flags_with_absl()

_cond = lambda fn: lambda *args: jax.lax.cond(jnp.array(True), fn, fn, *args)


class TransformedRefsTest(jtu.JaxTestCase):

  @parameterized.named_parameters(("jit", "jit"), ("scan", "scan"),
                                  ("remat", "remat"))
  def test_transformed_ref_in(self, transform):
    def scan_fn(fn):
      return lambda *args: jax.lax.scan(lambda _, __: (fn(*args), ()), None, (),
                                        length=1)[0]
    transform = {"jit": jax.jit, "scan": scan_fn, "remat": jax.remat}[transform]

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

  def test_transformed_ref_in_vmap(self):
    @jax.jit
    def f(xs):
      x_ref = jax.new_ref(jnp.zeros((7, 3)))
      x_ref_view = x_ref.at[1:6]

      def inner(x_ref_view, x):
        x_ref_view[...] += x

      jax.vmap(inner, in_axes=(0, 0))(x_ref_view, xs)
      return x_ref[...]

    xs = jnp.ones((5, 3))
    expected = jnp.zeros((7, 3)).at[1:6].add(xs)
    self.assertAllClose(f(xs), expected)

  @parameterized.named_parameters(("vmap", jax.vmap), ("remat", jax.remat), ("cond", _cond), ("jit", jax.jit))
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

  @parameterized.named_parameters(("false", False), ("true", True))
  def test_temp(self, pure_ref):
    """vmap/remat should be able to write to a TransformedRef (ref.at[slice])."""
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

  @parameterized.named_parameters(("vmap", jax.vmap), ("remat", jax.remat), ("cond", _cond), ("jit", jax.jit))
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

  @parameterized.named_parameters(("vmap", jax.vmap), ("remat", jax.remat), ("cond", _cond), ("jit", jax.jit))
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


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())