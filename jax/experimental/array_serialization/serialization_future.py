import asyncio
import threading

import jax
import numpy as np

from .new_api import PyTreeDef, PyTreeT, PathLike, PickleModule
from .new_api import async_load, async_save  # async routines
from .new_api import load_pytree  # synchronous routine for data overview
from .new_api import PermissivePyTreeSerialization, Path
from .new_api import _LEAF_DATA_DIR
from .new_api import _TYPE_ID_LEAF_DELIMITER, _TENSORSTORE_SUFFIX

__all__ = ["nonblocking_load", "nonblocking_save"]

serialize_pytree = PermissivePyTreeSerialization.serialize_pytree

class SerializationFuture:
  """Keeps track of saving/loading serialized data via:
    - self.done() - non-blocking check whether the underlying coroutine finished
    - self.result() - gets the result of the coroutine, raises error if not done
    - self.pytree - the property describing the data overview (short leaf desc.)
  
  The class takes in an async_fn and args/kwargs for it and launches it 
  immediately in a separate Python thread. This allows it to work both in sync 
  as well as in an async environment.
  """
  def __init__(self, async_fn, *args, **kw):
    self._pytree = None
    # create a closure which will run an asyncio routine in separate thread
    # and will populate either self._retval if no errors were raised or 
    # self._exception if there were errors

    # running an asyncio routine in a thread is a reliable way of scheduling
    # an asyncio routine in the background both in regular synchronous contexts
    # but also in (unexpectedly?) asynchronous contexts like Jupyter Notebooks
    self._retval, self._exception = [None], [None]

    def _run_in_thread():
      async def _async_thread_run():
        ret, exc = None, None
        try:
          ret = await async_fn(*args, **kw)
        except Exception as e:
          exc = e
        return ret, exc

      # populate either the result or the exception
      self._retval[0], self._exception[0] = asyncio.run(_async_thread_run())
    self._thread = threading.Thread(target=_run_in_thread)
    self._thread.start()
    # do not join the thread
    
  @property
  def pytree(self):
    return self._pytree

  @pytree.setter
  def pytree(self, new_pytree: PyTreeDef):
    msg = f"You cannot set the .pytree property in {type(self)} more than once."
    assert self._pytree is None or self._pytree == new_pytree, msg
    self._pytree = new_pytree
    
  def done(self):
    """Check if the underlying deserialization is done. Non-blocking."""
    return not self._thread.is_alive()

  def result(self):
    """Retrieve the result or raise an exception of the `async_fn`."""
    assert self.done()
    if self._exception[0] is not None:  # exception has been raised
      raise self._exception[0]
    return self._retval[0]

  def join(self):
    """Wait for the underlying thread to complete."""
    return self._thread.join()

def _construct_shape_dtype_pytree_from_leaf_ids(directory: str | PathLike, 
                                                pytree: PyTreeT | None = None):
  leaf_ids_flat, tree_struct = jax.tree_flatten(pytree)
  node_data_dir = (Path(directory) / _LEAF_DATA_DIR)
  assert node_data_dir.exists()
  paths = list(node_data_dir.iterdir())
  leaf_id2path = {path.stem: path for path in paths}
  
  print(leaf_ids_flat)

  leaf_desc_flat = []
  for leaf_id in leaf_ids_flat:
    leaf_id: str
    if _TYPE_ID_LEAF_DELIMITER in leaf_id:
      leaf_type, leaf_id = leaf_id.split(_TYPE_ID_LEAF_DELIMITER, 1)
    else:
      leaf_type, leaf_id = "unknown", leaf_id
    leaf_type = leaf_type.strip()
    assert leaf_id in leaf_id2path
    suffix = leaf_id2path[leaf_id].suffix
    if suffix == _TENSORSTORE_SUFFIX:
      # we're dealing with an array
      leaf_type = leaf_type.strip()
      # we're splitting a descriptor of the form `dtype[shape]`
      # e.g., float32[14, 2]
      dtype_str, shape_str = leaf_type.split("[", 1)
      shape = [int(x.strip()) for x in shape_str.strip("]").strip().split(",") 
               if len(x.strip()) > 0]
      dtype = jax.numpy.dtype(dtype_str)
      leaf_desc_flat.append(jax.ShapeDtypeStruct(shape, dtype))
    else:
      leaf_desc_flat.append(leaf_type) # TODO(rdyro): return actual type
  return jax.tree_unflatten(tree_struct, leaf_desc_flat)

    
def _pytree_leaf_desc(leaf):
  if isinstance(leaf, (np.ndarray, jax.Array)):
    return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
  else:
    return leaf
  
def nonblocking_save(data: PyTreeT, directory: str | PathLike, 
                     overwrite: bool = True, 
                     pickle_module: PickleModule | None = None,
                     ) -> SerializationFuture:
    # start serialization immediately
    fut = SerializationFuture(async_save, data, directory, overwrite, 
                              pickle_module)
    # construct a nice looking pytree representing the nodes being read
    fut.pytree = jax.tree_map(_pytree_leaf_desc, data)
    return fut
    
                     
def nonblocking_load(directory: str | PathLike, 
                     shardings: PyTreeT | None = None, 
                     pytree: PyTreeT | None = None,
                     pickle_module: PickleModule | None = None,
                     best_effort: bool = False) -> SerializationFuture:

  fut = SerializationFuture(async_load, directory, shardings, pytree, 
                            pickle_module, best_effort)
  # the user provided a pytree, we'll use this
  # TODO(rdyro): technically, the user is expected to provide a pytree of UUIDs
  if pytree is None: 
    pytree = load_pytree(directory, pickle_module, best_effort)
  fut.pytree = _construct_shape_dtype_pytree_from_leaf_ids(directory, pytree)
  return fut
