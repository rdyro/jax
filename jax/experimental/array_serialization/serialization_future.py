import asyncio
import threading

import jax
from jax._src.util import safe_zip

from .new_api import PyTreeDef, PyTreeT, PathLike, PickleModule
from .new_api import async_load, async_save  # async routines
from .new_api import load_pytree  # synchronous routine for data overview
from .new_api import PermissivePyTreeSerialization, _join_leaf_type_and_id
from .new_api import _TREE_REPR_KEY, _LEAF_IDS_KEY, _leaf_to_type_desc

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

  
def nonblocking_save(data: PyTreeT, directory: str | PathLike, 
                     overwrite: bool = True, 
                     pickle_module: PickleModule | None = None,
                     ) -> SerializationFuture:
    # start serialization immediately
    fut = SerializationFuture(async_save, data, directory, overwrite, 
                              pickle_module)
    # construct a nice looking pytree representing the nodes being read
    data_flat, pytreedef = jax.tree_util.tree_flatten(data)
    node_repr, leaf_ids, node_data_store = serialize_pytree(
      pytreedef, pickle_module)
    leaf_ids_flat = node_repr[_LEAF_IDS_KEY]
    inscribed_leafs_flat = {
      _join_leaf_type_and_id(_leaf_to_type_desc(leaf), leaf_id) 
      for (leaf_id, leaf) in safe_zip(leaf_ids_flat, data_flat)}
    fut.pytree = jax.tree_unflatten(pytreedef, inscribed_leafs_flat)

    return fut
                     
def nonblocking_load(directory: str | PathLike, 
                     shardings: PyTreeT | None = None, 
                     pytree: PyTreeT | None = None,
                     pickle_module: PickleModule | None = None,
                     best_effort: bool = False) -> SerializationFuture:

  if pytree is None:
    pytree = load_pytree(directory, pickle_module, best_effort)
  fut = SerializationFuture(async_load, directory, shardings, pytree, 
                            pickle_module, best_effort)
  fut.pytree = pytree
  return fut
