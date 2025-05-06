# Copyright 2025 The JAX Authors.
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

"""
Serializations routines for pytrees including array and non-array serialization.
"""

from __future__ import annotations

from os import PathLike
import os
import re
from typing import Any
from uuid import uuid4, UUID
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import shutil
import logging

import jax
from jax.tree_util import PyTreeDef
from jax._src.layout import Layout

from jax.experimental import multihost_utils
from jax.experimental.array_serialization import tensorstore_impl as ts_impl
import jax.experimental.array_serialization.pytree_serialization_utils as utils
from jax.sharding import SingleDeviceSharding
from jax._src.path import epath_installed, Path
import numpy as np

logger = logging.getLogger(__name__)

_THREADING_SAVE_LOCK = threading.Lock()

_REMOTE_URL_PREFIXES = ['gs://', 's3://']
_PYTREEDEF_FILE = "pytreedef.json"
_TENSORSTORE_SUFFIX = ".tensorstore"
_ARCHIVE_NAME = "archive.zip"
_USE_OCDBT = True  # a lot of the code relies on this being True
_MAX_PATH_LENGTH = 4096
_ARRAY_STORE_DIRNAME = f"array_store{_TENSORSTORE_SUFFIX}"
_ARRAY_TYPE_FORMAT = "Array({dtype}[{shape}])"
_ARRAY_TYPE_REGEX = r"Array\(([a-zA-Z0-9_]+)\[([0-9, ]*)\]\)"
_MAX_CONCURRENCY = 32

__all__ = ["save", "load", "load_pytreedef", "nonblocking_load",
           "nonblocking_save"]

PyTreeT = Any

def _get_unique_sync_key() -> str | None:
  """Generate a thread-local key for ensuring all host finish (de)serializing"""
  if jax.process_count() == 1:
    return None
  # broadcast a thread-local unique barrier name
  sync_key_id = UUID(bytes=np.array(multihost_utils.broadcast_one_to_all(
    np.frombuffer(uuid4().bytes, dtype=np.int32))).tobytes())
  sync_key = f"jax_sync_key_{str(sync_key_id)}"
  return sync_key

def _is_str_same_on_all_hosts(path: str | PathLike[str]) -> bool:
  """All-gather the location of the checkpoint and check if it's the same."""
  if jax.process_count() <= 1:
    return False
  path_b = str(path).encode("utf-8")
  assert len(path_b) <= _MAX_PATH_LENGTH, (
      f"Path exceeds maximum length of {_MAX_PATH_LENGTH} in multiprocess"
      " case.")
  path_array = np.concatenate([
      np.frombuffer(path_b, dtype=np.uint8), np.zeros(
          _MAX_PATH_LENGTH - len(path_b), dtype=np.uint8)])
  all_path_arrays = multihost_utils.process_allgather(path_array)
  return bool(np.all(all_path_arrays == all_path_arrays[:1, ...]))

def _sync_on_key(key: str | None, extra_tag: str = "") -> None:
  if key is None:
    return
  full_key = key if not extra_tag else f"{key}-{extra_tag}"
  multihost_utils.sync_global_devices(full_key)

def _is_array_like(x):
  return isinstance(x, (jax.Array, np.ndarray))

def _leaf_to_desc(leaf) -> str:
  if leaf is None:
    return "null"
  elif isinstance(leaf, (np.ndarray, jax.Array)):
    return _ARRAY_TYPE_FORMAT.format(
        dtype=leaf.dtype.name, shape=", ".join(map(str, leaf.shape)))
  else:
    return type(leaf).__name__

def _desc_to_leaf(leaf_desc: str) -> str | jax.ShapeDtypeStruct:
  if not re.match(_ARRAY_TYPE_REGEX, leaf_desc):
    return leaf_desc
  shape_dtype_match = re.match(_ARRAY_TYPE_REGEX, leaf_desc)
  assert shape_dtype_match is not None, (
      f"Failed to parse array descriptor: {leaf_desc} with pattern:"
      f" {_ARRAY_TYPE_REGEX}")
  dtype_str, shape_str = shape_dtype_match.groups()
  shape = [int(x.strip()) for x in shape_str.strip("]").strip().split(",")
            if len(x.strip()) > 0]
  return jax.ShapeDtypeStruct(shape, jax.numpy.dtype(dtype_str))

def _is_remote_path(path: str | PathLike[str]):
  """Check whether a path is remote by examining the prefix."""
  # we need to truncate e.g., gs:// to gs:/ because pathlib.Path collapses //
  return any(str(path).startswith(prefix[:-1])
             for prefix in _REMOTE_URL_PREFIXES)

def _norm_path(path: str | PathLike[str]) -> Path:
  if _is_remote_path(path):
    return Path(path)
  return Path(path).expanduser().resolve()

def _rm_dir(root: Path) -> None:  # pytype: disable=invalid-annotation
  if _is_remote_path(root):
    root.rmtree()  # pytype: disable=attribute-error
  else:
    shutil.rmtree(root)

def _set_up_destination(root: Path, overwrite: bool,  # pytype: disable=invalid-annotation
                        pytree_repr: dict[str, Any],
                        distinct_locations: bool, sync_key: str | None
                        ) -> dict[str, Any]:
  """Inspect the destination, set it up for writing, potentially read existing data."""
  root = _norm_path(root)
  if overwrite:
    if root.exists() and len(list(root.iterdir())) > 0:
      # check that we're only deleting things that come from JAX
      # refuse to rm directories containing additional entries
      extra_member_paths = [
          path for path in list(root.iterdir()) if path.name not in
          (_PYTREEDEF_FILE, _ARCHIVE_NAME, _ARRAY_STORE_DIRNAME)]

      assert len(extra_member_paths) == 0, (
        "Refusing to work on a directory that is not a previous checkpoint."
        f" Unrecognized paths: {extra_member_paths}. Remove them manually if"
        f" you're sure you want to use {root} as the checkpoint directory.")

      if (jax.process_index() == 0 or distinct_locations) and root.exists():
        _rm_dir(root)
    _sync_on_key(sync_key, "overwrite")
    return pytree_repr
  else:
    if (root.exists() and len(list(root.iterdir())) > 0):  # not empty
      raise ValueError(f"Files already exist at path: `{root}`, but you"
                       f" specified `{overwrite=}`")
    return pytree_repr

def _prepare_directory(root: Path, overwrite: bool,  # pytype: disable=invalid-annotation
                       pytreedef_repr: dict[str, Any],
                       distinct_locations: bool, sync_key: str | None):
  """Prepare the directory: check destination, potentially read existing data
  and overwrite.
  """
  root = _norm_path(root)
  # prepare the destination directory, overwrite destination directory or error
  pytreedef_repr = _set_up_destination(
      root, overwrite, pytreedef_repr, distinct_locations, sync_key)

  if not _is_remote_path(root):
    if distinct_locations or jax.process_index() == 0:
      root.mkdir(exist_ok=True)  # do not make parents, that's too much
      if not root.exists() or not root.is_dir():
        raise RuntimeError(f"Could not create destination directory at {root}")
  _sync_on_key(sync_key, "mkdir")
  return pytreedef_repr

async def _serialize_array(arr, path, extra_config, distinct_locations: bool
                          ) -> None:
  arr = jax.numpy.asarray(arr, dtype=arr.dtype)
  extra_ts_spec = extra_config
  process_idx = (jax.process_index() if (
      jax.process_count() > 1 and not distinct_locations) else None)

  default_ts_spec = ts_impl.get_tensorstore_spec(
      path, ocdbt=_USE_OCDBT, process_idx=process_idx, arr=arr)
  ts_spec = ts_impl.merge_nested_specs(default_ts_spec, extra_ts_spec)

  # verify the merged spec
  expected_path = default_ts_spec["kvstore"]["base"]["path"]
  ts_impl.verify_tensorstore_spec(ts_spec, arr, expected_path,
                                  check_metadata=True)

  # all hosts write because they're writing to different storage locations (to
  # be combined later) -> `primary_host=None`
  await ts_impl.async_serialize(arr, ts_spec, primary_host=None)

def _write_pytreedef(directory: Path, pytree_repr: dict[str, Any],  # pytype: disable=invalid-annotation
                     distinct_locations: bool):
  """Write the pytreedef to the desitination directory and aux data to the archive."""
  if not (jax.process_index() == 0 or distinct_locations):
    return
  root = _norm_path(directory)
  (root / _PYTREEDEF_FILE).write_text(json.dumps(pytree_repr, indent=2))

def _write_arrays(arrs_and_paths: list[tuple[Any, Path]],  # pytype: disable=invalid-annotation
                  full_ts_specs: list[Any | None],
                  distinct_locations: bool):

  async def _serialize_arrays():
    await asyncio.gather(*[_serialize_array(
      arr, path, extra_ts_spec, distinct_locations)
      for ((arr, path), extra_ts_spec)
      in zip(arrs_and_paths, full_ts_specs)])

  asyncio.run(_serialize_arrays())

def _finalize_array_store(kvstore_path, extra_config, distinct_locations: bool
                         ) -> None:
  """When multiple processes are writing, they must write to a per-processlocation
  followed by combining them via no-copy links to the final location.
  """
  # only in multiprocess case and only process 0
  if distinct_locations or jax.process_count() <= 1 or jax.process_index() != 0:
    return
  extra_ts_spec = extra_config
  dummy_key_path = os.path.join(kvstore_path, "dummy_key")
  combined_ts_spec = ts_impl.merge_nested_specs(ts_impl.get_tensorstore_spec(
      dummy_key_path, ocdbt=_USE_OCDBT, process_idx=None), extra_ts_spec)
  children_ts_spec = [ts_impl.merge_nested_specs(ts_impl.get_tensorstore_spec(
      dummy_key_path, ocdbt=_USE_OCDBT, process_idx=i), extra_ts_spec)
      for i in range(jax.process_count())]
  combined_kvstore = combined_ts_spec["kvstore"]
  children_kvstores = [ts_spec["kvstore"] for ts_spec in children_ts_spec]
  _ = combined_kvstore.pop("path")
  _ = [kvstore.pop("path") for kvstore in children_kvstores]
  asyncio.run(ts_impl.combine_kvstores(combined_kvstore, children_kvstores))

def save(data: PyTreeT, directory: str | PathLike[str], overwrite: bool = True,
         ts_specs: PyTreeT | None = None) -> None:
  """Saves the given data structure to the provided directory path.

  This function provides functionality to serialize and save a data structure
  comprising JAX arrays, along with its structure to a given directory. It
  leverages `PyTree` for flattening and reconstructing the data structure.

  .. code-block:: python

    data = {'a': jnp.array([1, 3]), 'b': {'c': [jnp.array([4, 5])]}}
    pytree_serialization.save(data, directory)

  Args:
    data: The data structure to be saved. Arbitrary composition of JAX arrays,
      including nested structures.
    directory: The directory path where the data will be saved. A local path or
      a remote URL (e.g., gs://, s3://). For remote URLs, `etils` is required.
    overwrite: If True, any existing directory with the same name will be
      overwritten.

  .. code-block:: python
    data = {"a": jnp.array([1, 2]), "b": None}
    save(data, directory)
  """
  with _THREADING_SAVE_LOCK:
    return _save(data, directory, overwrite, ts_specs)

def _save(data: PyTreeT, directory: str | PathLike[str], overwrite: bool = True,
         ts_specs: PyTreeT | None = None) -> None:
  sync_key = _get_unique_sync_key()  # get a synchronization key for multi-host

  assert not _is_remote_path(directory) or epath_installed, (
    "For saving to remote URLs (e.g., gs, s3) you need the `etils` module"
    "installed. You can install it using `pip install etils`.")
  data_flat, pytreedef = jax.tree.flatten(data, is_leaf=lambda x: x is None)
  distinct_locations = not _is_str_same_on_all_hosts(directory)
  if jax.process_count() > 1 and distinct_locations:
    logger.warning("Saving to different locations on different hosts is"
                   " supported, but extremely fragile. Consider using a single"
                   " location.")
  root = _norm_path(directory)

  # start serialization ##################################
  futures, executor = [], ThreadPoolExecutor(max_workers=_MAX_CONCURRENCY)

  # serialize the pytree #################################
  pytreedef_repr = utils.serialize_pytreedef(pytreedef)
  pytreedef_repr[utils._LEAF_IDS_KEY] = jax.tree.map(_leaf_to_desc, data_flat)
  leaf_ids_flat = list(range(len(pytreedef_repr[utils._LEAF_IDS_KEY])))

  pytreedef_repr = _prepare_directory(root, overwrite,
                                      pytreedef_repr, distinct_locations,
                                      sync_key)
  futures.append(executor.submit(_write_pytreedef, root, pytreedef_repr,
                                  distinct_locations))

  # serialize arrays #####################################
  array_store_path = root / _ARRAY_STORE_DIRNAME
  arrs_and_paths = [(data, array_store_path / str(leaf_id)) for data, leaf_id in
                    zip(data_flat, leaf_ids_flat)]
  full_ts_specs = (([None] * len(arrs_and_paths)) if ts_specs is None else
                   jax.tree.leaves(ts_specs, ts_impl.is_tensorstore_spec_leaf))
  futures.append(executor.submit(_write_arrays, arrs_and_paths, full_ts_specs,
                                 distinct_locations))

  # wait for all futures to complete #####################
  _ = [fut.result() for fut in futures]
  _sync_on_key(sync_key, "array_serialization")
  if len(arrs_and_paths) > 0:
    _finalize_array_store(array_store_path, full_ts_specs[0],
                          distinct_locations)
  # we are done with all async ops here, we can block ####
  _sync_on_key(sync_key, "end")

async def _deserialize_array(
    path: str | PathLike[str], sharding: jax.sharding.Sharding | Layout,
    ts_spec: dict[str, Any],
    byte_limiter: ts_impl._LimitInFlightBytes | None = None) -> jax.Array:
  """Deserialize an array from a given path with an optional extra tensorstore spec."""
  # every process reads from the central location
  default_ts_spec = ts_impl.get_tensorstore_spec(
      path, ocdbt=_USE_OCDBT, process_idx=None)
  expected_path = default_ts_spec['kvstore']['base']['path']
  ts_spec = ts_impl.merge_nested_specs(default_ts_spec, ts_spec)
  ts_impl.verify_tensorstore_spec(ts_spec, arr=None, path=expected_path,
                                  check_metadata=False)
  return await ts_impl.async_deserialize(sharding, ts_spec,
                                         byte_limiter=byte_limiter)

def _read_arrays(array_store_path: Path, arr_leaf_ids: list[int],  # pytype: disable=invalid-annotation
                 ts_specs: Any | None,
                 shardings: PyTreeT | utils._MISSING_TYPE):
  # array_store_path = root / _LEAF_DATA_DIR / _ARRAY_STORE_DIRNAME
  arr_store_path = _norm_path(array_store_path)
  arr_paths = [arr_store_path / str(leaf_id) for leaf_id in arr_leaf_ids]
  # missing sharding assumes we want to deserialize on default device
  if shardings is utils.MISSING:
    device = jax.devices()[0]  # default device
    shardings = [SingleDeviceSharding(device) for _ in arr_paths]
  else:
    shardings = jax.tree.flatten(shardings, is_leaf=lambda x: x is None)[0]
    assert len(shardings) == len(arr_paths), (
      "The sharding leaves must match the load arrays requested.")
  full_ts_specs = (([None] * len(arr_paths))
                   if ts_specs is None else jax.tree.leaves(
                       ts_specs, is_leaf=ts_impl.is_tensorstore_spec_leaf))
  # byte limiter to limit number of parallel reads, resizes to largest read
  byte_limiter = ts_impl._LimitInFlightBytes(10 * 1024 ** 3)  # 10 GB

  async def _deserialize_arrays():
    return await asyncio.gather(*[
        _deserialize_array(path, sharding, ts_spec, byte_limiter)
        for (path, sharding, ts_spec)
        in zip(arr_paths, shardings, full_ts_specs)])

  arr_keys = [int(path.stem) for path in arr_paths]

  # finally, collect the results
  return dict(zip(arr_keys, asyncio.run(_deserialize_arrays())))

def load_pytreedef(directory: str | PathLike[str]) -> PyTreeDef:
  """Loads a pytree from the given directory.
  Args:
    directory: Directory path to load from.
  Returns:
    The loaded pytree with arrays represented as jax.ShapeDtypeStruct's.
  """
  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`.")
  json_content = (_norm_path(directory) / _PYTREEDEF_FILE).read_text()
  raw_tree = json.loads(json_content)
  leaves = map(_desc_to_leaf, raw_tree[utils._LEAF_IDS_KEY])
  return jax.tree.unflatten(utils.deserialize_pytreedef(raw_tree), leaves)

_prefix_mask = lambda m, x: jax.tree.map(lambda _: None, x) if not m else x

def load(directory: str | PathLike[str],
         shardings: PyTreeT | utils._MISSING_TYPE = utils.MISSING,
         mask: PyTreeT | utils._MISSING_TYPE = utils.MISSING,
         ts_specs: PyTreeT | None = None) -> PyTreeT:
  """Loads and reconstructs a data structure from a directory.

  Args:
    directory: Directory path where the data is stored.
    shardings: Sharding strategy for array objects. If None, defaults to
      single device sharding on the default device.
  Returns:
    Reconstructed data structure.

  .. code-block:: python
    data = load(directory)
    # deserialize ignoring unknonwn nodes (return custom nodes as a list of children)
    data = load(directory)
  """
  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`.")

  root = _norm_path(directory)
  assert root.is_dir(), f"Checkpoint directory {root} does not exist"

  # deserialize PyTreeDef
  pytree = load_pytreedef(directory)
  if mask is not utils.MISSING:
    pytree = jax.tree.map(_prefix_mask, mask, pytree)
  pytreedef = jax.tree.structure(pytree, is_leaf=lambda x: x is None)
  leaf_ids_flat = jax.tree.leaves(pytree, is_leaf=lambda x: x is None)
  executor = ThreadPoolExecutor(max_workers=_MAX_CONCURRENCY)
    
  # deserialize array objects
  arr_leaf_ids = [i for i, leaf_id in enumerate(leaf_ids_flat)
                  if leaf_id is not None]
  arrs_fut = executor.submit(_read_arrays, root / _ARRAY_STORE_DIRNAME,
                             arr_leaf_ids, ts_specs, shardings)

  arrs = arrs_fut.result()
  filled_values = [arrs.get(i, None) for i, _ in enumerate(leaf_ids_flat)]
  return jax.tree.unflatten(pytreedef, filled_values)

nonblocking_executor = ThreadPoolExecutor(max_workers=_MAX_CONCURRENCY)

def nonblocking_save(data: PyTreeT, directory: str | PathLike[str],
                     overwrite: bool = True,
                     tensorstore_specs: PyTreeT | None = None,
                     ) -> utils.AwaitableFuture:
  """Start the serialization without blocking, return a
  concurrent.futures.Future future to the result.

  .. code-block:: python
    fut = nonblocking_save(data, directory)
    print(fut.pytree)  # a pytree of jax.ShapeDtypeStruct's
    print(fut.result())  # None, blocking until the serialization is done
  """
  # start serialization immediately
  fut = utils.AwaitableFuture(nonblocking_executor.submit(
      save, data, directory, overwrite, tensorstore_specs))
  # construct a nice looking pytree representing the nodes being read
  fut.pytree = jax.tree.map(lambda leaf:jax.typeof(leaf)
                            if _is_array_like(leaf) else leaf, data)
  return fut

def nonblocking_load(directory: str | PathLike[str],
                     shardings: PyTreeT | utils._MISSING_TYPE = utils.MISSING,
                     pytreedef: PyTreeT | utils._MISSING_TYPE = utils.MISSING,
                     tensorstore_specs: PyTreeT | None = None,
                     ) -> utils.AwaitableFuture:
  """Start deserialization without blocking, return a
  concurrent.futures.Future future to the result with a pytree stub

  .. code-block:: python
    fut = nonblocking_load(directory)
    print(fut.pytree)  # a pytree of jax.ShapeDtypeStruct
    print(fut.result())  # the fully populated pytree
  """
  if pytreedef is utils.MISSING:
    pytreedef = load_pytreedef(directory)

  # TODO(rdyro): the awaitable future output is a workaround
  # it should return the fully populated pytree instead of just
  # jax.ShapeDtypeStruct for arrays by constructing them asynchronously
  fut = utils.AwaitableFuture(nonblocking_executor.submit(
      load, directory, shardings, pytreedef, tensorstore_specs))
  fut.pytree = jax.tree.map(lambda leaf:jax.typeof(leaf)
                            if _is_array_like(leaf) else leaf, pytreedef)
  return fut
