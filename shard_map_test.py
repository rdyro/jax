from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, PartitionSpec as P, NamedSharding

jax.config.update("jax_num_cpu_devices", 2)

mesh = jax.make_mesh((jax.device_count(),), ("x",),
                     #axis_types=(AxisType.Auto,))
                     axis_types=(AxisType.Explicit,))
in_shardings = (P(None, "x"), P(None, "x", None))
out_shardings = P(None, "x")

in_shardings = tuple(NamedSharding(mesh, p) for p in in_shardings)
out_shardings = NamedSharding(mesh, out_shardings)


@partial(jax.jit, in_shardings=in_shardings, out_shardings=out_shardings)
def tp(x, A):
  gs = jnp.array([x.shape[0]], dtype=jnp.int32)
  return jax.lax.ragged_dot(x, A, gs)

x = jnp.ones((32, 128))
A = jnp.ones((1, 128, 256))

x, A = jax.device_put((x, A), in_shardings)

y = tp(x, A)
print(y)
