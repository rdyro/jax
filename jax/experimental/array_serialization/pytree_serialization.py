# Copyright 2021 The JAX Authors.
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

from __future__ import annotations

from os import PathLike
import os
from types import ModuleType
import re
from typing import Any
from uuid import uuid4, UUID
import json
import asyncio
import threading
import shutil
import pickle
import logging

import jax
from jax.tree_util import PyTreeDef
from jax.util import safe_zip
from jax._src import distributed
from jax._src.layout import Layout
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import asyncio_utils
from jax.experimental.array_serialization import tensorstore_impl as ts_impl
from jax.experimental.array_serialization.pytree_serialization_utils import (
    PermissivePyTreeSerialization, StrictPyTreeSerialization, InMemoryZip,
    _LEAF_IDS_KEY, _TREE_REPR_KEY)
from jax.sharding import SingleDeviceSharding
from jax._src.path import epath_installed, Path
import numpy as np

_REMOTE_URL_PREFIXES = ['gs://', 's3://']
_PYTREEDEF_FILE = "pytreedef.json"
_TENSORSTORE_SUFFIX = ".tensorstore"
_LEAF_DATA_DIR = "leaf_data"
_OBJ_DATA_ARCHIVE = "obj_data.zip"
_NODE_DATA_ARCHIVE = "node_data.zip"
_TYPE_ID_LEAF_DELIMITER = " -> "
_USE_OCDBT = True  # a lot of the code relies on this being True
_SYNC_KEY_TIMEOUT_MS = multihost_utils._TIMEOUT_MS
_MAX_PATH_LENGTH = 4096
_ERROR_ON_EXTRA_LEAVES = False
_ARRAY_STORE_DIRNAME = f"array_store{_TENSORSTORE_SUFFIX}"
_ARRAY_TYPE_NAME = "Array"

__all__ = ["save", "load", "load_pytree", "async_save",
           "async_load", "async_load_pytree",
           "nonblocking_load", "nonblocking_save"]

PyTreeT = Any
PickleModule = ModuleType

logger = logging.getLogger(__name__)

def _get_sync_client() -> distributed.xla_extension.DistributedRuntimeClient:
  assert jax.process_count() > 1, (
    "You are attempting to wait for other hosts, but there is only 1 host"
  )
  assert distributed.global_state.client is not None, (
    "The distributed runtime is not initialized. You likely need to call"
    " `jax.distributed.initialize()` first."
  )
  return distributed.global_state.client

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
  _get_sync_client().wait_at_barrier(full_key, _SYNC_KEY_TIMEOUT_MS, None)

def _is_array_like(x: Any):
  return isinstance(x, (jax.Array, np.ndarray))

def _get_suffix(x: Any):
  if _is_array_like(x):
    return _TENSORSTORE_SUFFIX
  elif isinstance(x, bytes):
    return ".bin"
  else:
    return ".json"

# pytype: disable=bad-return-type
async def _obj_serialize(archive: InMemoryZip, filename: str, x: Any,
                         pickle_module: PickleModule | None = None) -> str:
  """Serialization method for NOT-array objects."""
  # we're only interested in name and suffix
  filename = Path(Path(filename).name)
  if _is_array_like(x):
    raise ValueError
  elif isinstance(x, bytes):
    await archive.async_write(filename.with_suffix(".bin"), x)
    return ".bin"
  elif Path(filename).suffix == ".pickle":
    assert pickle_module is not None
    await archive.async_write(filename.with_suffix(".pickle"),
                              pickle_module.dumps(x))
    return ".pickle"
  else:  # attempt to write JSON - the default
    try:
      payload = json.dumps(x)
      await archive.async_write(filename.with_suffix(".json"), payload)
      return ".json"
    except TypeError as exc:  # object is not JSON serializable
      if not pickle_module:
        raise ValueError(
          "An object cannot be serialized using JSON, your pytree contains a"
          f" non-serializable object: {x}. Consider using"
          " `pickle_module=pickle`") from exc
      else:
        await archive.async_write(filename.with_suffix(".pickle"),
                                  pickle_module.dumps(x))
        return ".pickle"
# pytype: enable=bad-return-type

async def _obj_deserialize(
    archive: InMemoryZip, filename: str,
    pickle_module: PickleModule | None = None, best_effort: bool = False):
  """Deserialization method for NON-array objects."""
  filename = Path(filename)
  suffix = filename.suffix
  if _is_array_like(_TENSORSTORE_SUFFIX):
    raise ValueError
  elif suffix == ".bin":
    return await archive.async_read(filename)
  elif suffix == ".json":
    return json.loads(await archive.async_read(filename))
  elif suffix == ".pickle":
    if not pickle_module and not best_effort:
      raise ValueError(f"Pickle file encountered, but {pickle_module=}")
    raw_bytes = await archive.async_read(filename)
    if not pickle_module and best_effort:
      return raw_bytes
    try:
      assert hasattr(pickle_module, "loads")
      return pickle_module.loads(raw_bytes)
    except (pickle.PickleError, AttributeError, AssertionError) as exc:
      raise ValueError(f"We couldn't deserialize obj at `{filename.name}` using"
                       f" {pickle_module=}") from exc
  else:
    raise ValueError(f"Suffix `{suffix}` deserialization is not supported.")

def _leaf_to_type_desc(leaf) -> str:
  if leaf is None:
    return "null"
  elif isinstance(leaf, (np.ndarray, jax.Array)):
    return (f"{_ARRAY_TYPE_NAME}[[{', '.join(map(str, leaf.shape))}]," +
            f" {leaf.dtype.name}]")
  else:
    return type(leaf).__name__

def _leaf_desc_to_leaf(leaf_desc: str) -> str | jax.ShapeDtypeStruct:
  leaf_type: str = (leaf_desc.split(_TYPE_ID_LEAF_DELIMITER, 1)[0]
                    if _TYPE_ID_LEAF_DELIMITER in leaf_desc else leaf_desc)
  if not leaf_type.startswith(_ARRAY_TYPE_NAME):
    return leaf_type
  pat = r"Array\[\[([0-9, ]*)\],\s*([a-zA-Z0-9_]+)\]"
  shape_dtype_match = re.match(pat, leaf_type)
  assert shape_dtype_match is not None, (
      f"Failed to parse array descriptor: {leaf_type} with pattern: {pat}")
  shape_str, dtype_str = shape_dtype_match.groups()
  shape = [int(x.strip()) for x in shape_str.strip("]").strip().split(",")
            if len(x.strip()) > 0]
  dtype = jax.numpy.dtype(dtype_str)
  return jax.ShapeDtypeStruct(shape, dtype)

def _join_leaf_type_and_id(leaf_type: str, leaf_id: str) -> str:
  return f"{leaf_type}{_TYPE_ID_LEAF_DELIMITER}{leaf_id}"

def _inscribe_leaf_types(pytree_repr: dict[str, Any],
                         leaf_id_type_map: dict[str, str]):
  """Rewrite a JSON PyTree representation by adding type to leaf_id."""
  if pytree_repr["node_type"] == "leaf":
    leaf_id = pytree_repr["leaf_id"]
    pytree_repr["leaf_id"] = _join_leaf_type_and_id(leaf_id_type_map[leaf_id],
                                                    leaf_id)
  else:
    _ = [_inscribe_leaf_types(child, leaf_id_type_map)
         for child in pytree_repr["children"]]

def _is_remote_path(path: str | PathLike[str]):
  # we check whether a path is remote by checking the prefix
  # we need to truncate e.g., gs:// to gs:/ because pathlib.Path collapses //
  return any(str(path).startswith(prefix[:-1])
             for prefix in _REMOTE_URL_PREFIXES)

async def serialize_array(arr, path, extra_config,
                          distinct_locations: bool) -> None:
  arr = jax.numpy.asarray(arr, dtype=arr.dtype)
  extra_ts_spec = extra_config
  process_num = (jax.process_index() if (
      jax.process_count() > 1 and not distinct_locations) else None)
  default_ts_spec = ts_impl.get_tensorstore_spec(
      path, ocdbt=_USE_OCDBT, process_num=process_num, arr=arr)
  expected_path = default_ts_spec['kvstore']['base']['path']
  ts_spec = ts_impl.merge_nested_specs(default_ts_spec, extra_ts_spec)
  ts_impl.verify_tensorstore_spec(ts_spec, arr, expected_path,
                                  check_metadata=True)
  # all hosts write because they're writing to different storage locations (to
  # be combined later) -> `primary_host=None`
  await ts_impl.async_serialize(arr, ts_spec, primary_host=None)

async def finalize_array_store(kvstore_path, extra_config,
                               distinct_locations: bool) -> None:
  # only in multiprocess case and only process 0
  if distinct_locations or jax.process_count() <= 1 or jax.process_index() != 0:
    return
  extra_ts_spec = extra_config
  dummy_key_path = os.path.join(kvstore_path, "dummy_key")
  combined_ts_spec = ts_impl.merge_nested_specs(ts_impl.get_tensorstore_spec(
      dummy_key_path, ocdbt=_USE_OCDBT, process_num=None), extra_ts_spec)
  children_ts_spec = [ts_impl.merge_nested_specs(ts_impl.get_tensorstore_spec(
      dummy_key_path, ocdbt=_USE_OCDBT, process_num=i), extra_ts_spec)
      for i in range(jax.process_count())]
  combined_kvstore = combined_ts_spec["kvstore"]
  children_kvstores = [ts_spec["kvstore"] for ts_spec in children_ts_spec]
  _ = combined_kvstore.pop("path")
  _ = [kvstore.pop("path") for kvstore in children_kvstores]
  await ts_impl.combine_kvstores(combined_kvstore, children_kvstores)

async def deserialize_array(
    path: str | PathLike[str], sharding: jax.sharding.Sharding | Layout,
    ts_spec: dict[str, Any],
    byte_limiter: asyncio_utils.LimitInFlightBytes | None = None) -> jax.Array:
  # every process reads from the central location
  default_ts_spec = ts_impl.get_tensorstore_spec(
      path, ocdbt=_USE_OCDBT, process_num=None)
  expected_path = default_ts_spec['kvstore']['base']['path']
  ts_spec = ts_impl.merge_nested_specs(default_ts_spec, ts_spec)
  ts_impl.verify_tensorstore_spec(ts_spec, arr=None, path=expected_path,
                                  check_metadata=False)
  return await ts_impl.async_deserialize(
      sharding, ts_spec, byte_limiter=byte_limiter)


async def async_save(data: PyTreeT, directory: str | PathLike[str],
                     overwrite: bool = True,
                     pickle_module: PickleModule | None = None,
                     ts_specs: PyTreeT | None = None) -> None:
  """Saves the given data structure to the provided directory path.

  This function provides functionality to serialize and save a data structure
  comprising JAX arrays, NumPy arrays, Python objects, etc., along with its
  structure to a given directory. It leverages `PyTree` for flattening and
  reconstructing the data structure.

  If `pickle_module` is provided, we use pickle to serialize unknown types.

  Args:
    data: The data structure to be saved. Arbitrary composition of JAX arrays,
      NumPy arrays, and Python objects, including nested structures.
    directory: The directory path where the data will be saved. A local path or
      a remote URL (e.g., gs://, s3://). For remote URLs, `etils` is required.
    overwrite: If True, any existing directory with the same name will be
      overwritten.
    pickle_module: If not None, uses a pickling serialization strategy that uses
      the provided pickle_module to serialize any unsupported types.
  Raises:
    AssertionError: If attempting to save to a remote path without the `etils`
      package installed.
    NotImplementedError: If `overwrite` is False and a checkpoint already
      exists at the provided directory.
  """
  sync_key = _get_unique_sync_key()  # get a synchronization key for multi-host

  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`."
  )
  data_flat, pytreedef = jax.tree.flatten(data)
  serialize_pytree = PermissivePyTreeSerialization.serialize_pytree
  distinct_locations = not _is_str_same_on_all_hosts(directory)
  if jax.process_count() > 1 and distinct_locations:
    logger.warning("Saving to different locations on different hosts is"
                   " supported, but extremely fragile. Consider using a single"
                   " location.")

  # overwrite or error
  root = Path(directory)
  if overwrite:
    if root.exists() and len(list(root.iterdir())) > 0:
      # check that we're only deleting things that come from JAX
      # refuse to rm directories containing additional entries
      paths_present = list(Path(directory).iterdir())
      extra_member_paths = [path for path in paths_present if path.name not in
                            (_PYTREEDEF_FILE, _LEAF_DATA_DIR,
                             _NODE_DATA_ARCHIVE)]
      assert len(extra_member_paths) == 0, (
        "Refusing to remove a directory that is not a previous checkpoint."
        f" Unrecognized paths: {extra_member_paths}. Remove them manually if"
        f" you're sure you want to use {root} as the checkpoint directory.")
      if Path(directory).exists() and (jax.process_index() == 0
                                       or distinct_locations):
        if _is_remote_path(directory):
          Path(directory).rmtree()  # pytype: disable=attribute-error
        else:
          shutil.rmtree(Path(directory))
    _sync_on_key(sync_key, "overwrite")
  else:
    if (root.exists() and len(list(root.iterdir())) > 0):  # not empty
      raise NotImplementedError(f"Files already exist at path: `{root}`,"
                                f" but you specified `{overwrite=}`")

  if not _is_remote_path(directory):
    if distinct_locations:
      root = root.resolve()
      root.mkdir(exist_ok=True)  # do not make parents, that's too much
      assert root.exists() and root.is_dir()
    else:
      if jax.process_index() == 0:
        root = root.resolve()
        root.mkdir(exist_ok=True)  # do not make parents, that's too much
        assert root.exists() and root.is_dir()
      _sync_on_key(sync_key, "mkdir")

  pytree_repr, leaf_ids, node_data_store = serialize_pytree(pytreedef,
                                                            pickle_module)
  leaf_ids_flat = jax.tree.flatten(leaf_ids)[0]
  # inscribe types into leaf ids in-place
  leaf_ids_type_map = {leaf_id: _leaf_to_type_desc(leaf) for (leaf_id, leaf)
                       in safe_zip(leaf_ids_flat, data_flat)}
  _inscribe_leaf_types(pytree_repr[_TREE_REPR_KEY], leaf_ids_type_map)
  pytree_repr[_LEAF_IDS_KEY] = [
    _join_leaf_type_and_id(leaf_ids_type_map[leaf_id], leaf_id)
    for leaf_id in leaf_ids_flat]

  # start serialization ##################################
  serialize_futures = []

  # 0. serialize the pytree
  # if not _is_remote_path(root) or jax.process_index() == 0:
  if jax.process_index() == 0 or distinct_locations:
    async def _write_pytree():
      (root / _PYTREEDEF_FILE).write_text(json.dumps(pytree_repr, indent=2))
    serialize_futures.append(_write_pytree())
  paths = [(root / _LEAF_DATA_DIR / f"{leaf_id}{_get_suffix(data)}")
            for leaf_id, data in safe_zip(leaf_ids_flat, data_flat)]

  # 1. serialize JSON serializable leafs
  obj_archive = InMemoryZip(read_mode=False)
  obj_serialize_futures, obj_paths = [], []  # pylint: disable=unused-variable
  if jax.process_index() == 0 or distinct_locations:
    obj_flat = [data for path, data in safe_zip(paths, data_flat)
                if path.suffix != _TENSORSTORE_SUFFIX]
    obj_paths = [path for path in paths if path.suffix != _TENSORSTORE_SUFFIX]
    for data, path in safe_zip(obj_flat, obj_paths):
      # each host has to read all binary files
      obj_serialize_futures.append(_obj_serialize(
          obj_archive, path.name, data, pickle_module=pickle_module))

  async def _write_objects():
    archive_path = (Path(directory) / _LEAF_DATA_DIR / _OBJ_DATA_ARCHIVE)
    archive_path.parent.mkdir(exist_ok=True)
    await asyncio.gather(*obj_serialize_futures)
    archive_path.write_bytes(await obj_archive.async_tobytes())

  # 3. serialize arrays
  arr_flat = [data for path, data in safe_zip(paths, data_flat)
              if path.suffix == _TENSORSTORE_SUFFIX]
  arr_paths = [path for path in paths if path.suffix == _TENSORSTORE_SUFFIX]
  arr_paths = [path.parent / _ARRAY_STORE_DIRNAME / path.stem  # no suffix
               for path in arr_paths]

  # primary host set to host 0 or None (all hosts write everything)
  ts_specs = (([None] * len(arr_flat)) if ts_specs is None else
              jax.tree.leaves(ts_specs,
                              is_leaf=ts_impl.is_tensorstore_spec_leaf))
  serialize_futures.extend([
      serialize_array(arr, path, extra_ts_spec, distinct_locations)
      for (arr, path, extra_ts_spec) in safe_zip(arr_flat, arr_paths, ts_specs)])

  # 3. serialize node data if permissive
  node_data_serialize_futures = []
  node_data_archive = InMemoryZip(read_mode=False)
  if pickle_module is not None and (not _is_remote_path(root) or
                                    jax.process_index() == 0):
    for node_data_key, node_data_bytes in node_data_store.items():
      node_data_serialize_futures.append(
          _obj_serialize(node_data_archive, node_data_key, node_data_bytes,
                         pickle_module=pickle_module))

  async def _write_node_data():
    if len(node_data_serialize_futures) > 0:
      archive_path = (Path(directory) / _NODE_DATA_ARCHIVE)
      await asyncio.gather(*node_data_serialize_futures)
      archive_path.write_bytes(await node_data_archive.async_tobytes())

  await asyncio.gather(asyncio.gather(*serialize_futures), _write_objects(),
                       _write_node_data())
  _sync_on_key(sync_key, "serialization")
  if len(arr_paths) > 0:
    await finalize_array_store(arr_paths[0].parent, ts_specs[0],
                               distinct_locations)
  # we are done with all async ops here, we can block
  _sync_on_key(sync_key, "end")

async def async_load(directory: str | PathLike[str],
                     shardings: PyTreeT | None = None,
                     pytree: PyTreeT | None = None,
                     pickle_module: PickleModule | None = None,
                     ts_specs: PyTreeT | None = None,
                     best_effort: bool = False) -> PyTreeT:
  """Loads and reconstructs a data structure from a directory.

  Args:
    directory: Directory path where the data is stored.
    shardings: Sharding strategy for array objects. If None, defaults to
      single device sharding on the default device.
    pytree: Optional pre-populated PyTree for structure. If provided, must
      specify a pytree with string object ids. Useful for partial reads.
    pickle_module: Pickle module supporting dumps and loads methods.
    best_effort: Proceed with deserialization even in the face of partial
      failures. Return custom nodes as a list of children.
  Returns:
    Reconstructed data structure.
  Raises:
    AssertionError: If attempting to load from a remote path without etils
      installed.
    ValueError: If data for specific leaf IDs is missing in the directory.
    ImportError: If supported node type (e.g., flax's FrozenDict) cannot be
      imported.
  """
  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`.")

  sync_key = _get_unique_sync_key()  # get a synchronization key for multi-host

  root = Path(directory)
  assert root.is_dir(), f"Checkpoint directory {root} does not exist"
  if not _is_remote_path(root):
    root = root.resolve()

  # deserialize in 3 stages

  # 1. deserialize PyTree (if permissive inserting node_data)
  if pytree is None:
    pytreedef = await async_load_pytree(directory, pickle_module, best_effort)
    leaf_ids, pytreedef = jax.tree.flatten(pytreedef)
  else:
    leaf_ids, pytreedef = jax.tree.flatten(pytree)
  obj_leaf_ids = [leaf_id.split(_TYPE_ID_LEAF_DELIMITER, 1)[1] for leaf_id
                  in leaf_ids if not leaf_id.startswith(_ARRAY_TYPE_NAME)]

  # 2. deserialize non-array objects
  async def _deserialize_objs():
    obj_archive = InMemoryZip(data=(root / _LEAF_DATA_DIR / _OBJ_DATA_ARCHIVE
                                    ).read_bytes())
    _key2id = lambda x: Path(x).stem
    obj_keys = list(obj_archive.keys())
    missing_leaf_ids = set(obj_leaf_ids) - set(map(_key2id, obj_keys))
    requested_obj_keys = [obj_key for obj_key in obj_keys
                          if _key2id(obj_key) in obj_leaf_ids]
    if len(missing_leaf_ids) > 0:
      raise ValueError(
        f"Values {missing_leaf_ids} are missing from the checkpoint directory.")
    obj_values = await asyncio.gather(*[_obj_deserialize(
        obj_archive, obj_key, pickle_module, best_effort)
                                        for obj_key in requested_obj_keys])
    return dict(safe_zip(map(_key2id, requested_obj_keys), obj_values))

  # 3. deserialize array objects
  arr_leaf_ids = [leaf_id.split(_TYPE_ID_LEAF_DELIMITER, 1)[1] for leaf_id
                  in leaf_ids if leaf_id.startswith(_ARRAY_TYPE_NAME)]
  arr_paths = [root / _LEAF_DATA_DIR / _ARRAY_STORE_DIRNAME / leaf_id
               for leaf_id in arr_leaf_ids]
  # missing sharding assumes we want to deserialize on default device
  if shardings is None:
    device = jax.devices()[0]  # default device
    shardings = [SingleDeviceSharding(device) for _ in arr_paths]
  else:
    shardings = jax.tree.flatten(shardings)[0]
  ts_specs = (([None] * len(arr_paths)) if ts_specs is None else
              jax.tree.leaves(ts_specs,
                              is_leaf=ts_impl.is_tensorstore_spec_leaf))
  byte_limiter = asyncio_utils.LimitInFlightBytes(100 * 1024 ** 3)  # 100 GB
  arrs = {path.stem: deserialize_array(path, sharding, ts_spec, byte_limiter)
          for (path, sharding, ts_spec)
          in safe_zip(arr_paths, shardings, ts_specs)}

  # finally, given all deserialize coroutines scheduled, await the results
  leaf_uuids = [leaf_id.split(_TYPE_ID_LEAF_DELIMITER, 1)[1]
                for leaf_id in leaf_ids]
  objs, arrs_values = await asyncio.gather(_deserialize_objs(),
                                           asyncio.gather(*arrs.values()))
  arr_and_objs = dict(safe_zip(arrs.keys(), arrs_values)) | objs
  filled_values = [arr_and_objs[leaf_id] for leaf_id in leaf_uuids]

  _sync_on_key(sync_key)  # we are done with all async ops here, we can block
  return jax.tree.unflatten(pytreedef, filled_values)


async def async_load_pytree(directory: str | PathLike[str],
                            pickle_module: PickleModule | None = None,
                            best_effort: bool = False) -> PyTreeDef:
  """Loads a pytree from the given directory.
  Args:
    directory: Directory path to load from.
    pickle_module: Pickle module supporting dumps and loads methods.
    best_effort: Proceed with deserialization even in the face of partial
      failures. Return custom nodes as a list of children.
  Returns:
    The loaded pytree.
  """
  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`.")
  root = Path(directory)
  json_content = (root / _PYTREEDEF_FILE).read_text()
  raw_tree = json.loads(json_content)

  async def _read_node_data_store():
    if (root / _NODE_DATA_ARCHIVE).exists():
      node_data_archive = InMemoryZip(data=(root / _NODE_DATA_ARCHIVE
                                           ).read_bytes())
      _key2id = lambda x: Path(x).stem
      node_data_keys = list(node_data_archive.keys())
      node_data_values = await asyncio.gather(*[
          node_data_archive.async_read(node_data_key)
          for node_data_key in node_data_keys])
      return dict(safe_zip(map(_key2id, node_data_keys), node_data_values))
    else:
      return None

  # detect if the written tree is permissive or strict
  if (isinstance(raw_tree, dict) and len(raw_tree) == 2
      and all(key in raw_tree for key in [_TREE_REPR_KEY, _LEAF_IDS_KEY])):
    # the checkpoint is permissive, ostensibly it needs a pickler to load binary
    # data, but the user might have used permissive mode to save
    # a strict / clean checkpoint, error lazily
    node_data_store = await _read_node_data_store()
    if (pickle_module is None and not best_effort
        and node_data_store is not None):
      raise ValueError(
        f"{_NODE_DATA_ARCHIVE} is exists, but pickle_module is not provided and"
        f" `{best_effort=}`. We cannot proceed.")
    node_data_store = node_data_store if node_data_store is not None else dict()
    return PermissivePyTreeSerialization.deserialize_pytree(
      raw_tree, node_data_store, pickle_module, best_effort)
  else:
    # the checkpoint is strict / not permissive
    return StrictPyTreeSerialization.deserialize_pytree(raw_tree)


################################################################################

load_pytree = asyncio_utils._maybe_run_async_sync("load_pytree",
                                                  async_load_pytree)
save = asyncio_utils._maybe_run_async_sync("save", async_save)
load = asyncio_utils._maybe_run_async_sync("load", async_load)

################################################################################

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
    self._retval, self._exception = None, None
    self._done_event = threading.Event()
    self.async_fn = async_fn

    def _run_in_thread():
      async def _async_thread_run():
        ret, exc = None, None
        try:
          ret = await self.async_fn(*args, **kw)
        except Exception as e:  # pylint: disable=broad-except
          exc = e
        return ret, exc
      # populate either the result or the exception
      self._retval, self._exception = asyncio.run(_async_thread_run())
      self._done_event.set()
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
    # return not self._thread.is_alive()
    return self._done_event.is_set()

  def result(self):
    """Retrieve the result or raise an exception of the `async_fn`."""
    assert self.done()
    self._thread.join()
    if self._exception is not None:  # exception has been raised
      raise self._exception
    return self._retval

  def __await__(self):
    while not self.done():
      yield
    self._thread.join()
    return self.result()

  def join(self):
    """Wait for the underlying thread to complete."""
    return self._thread.join()

def _pytree_leaf_desc(leaf):
  if isinstance(leaf, (np.ndarray, jax.Array)):
    return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
  else:
    return leaf

def nonblocking_save(data: PyTreeT, directory: str | PathLike[str],
                     overwrite: bool = True,
                     pickle_module: PickleModule | None = None,
                     tensorstore_specs: PyTreeT | None = None,
                     ) -> SerializationFuture:
  # start serialization immediately
  fut = SerializationFuture(async_save, data, directory, overwrite,
                            pickle_module, tensorstore_specs)
  # construct a nice looking pytree representing the nodes being read
  fut.pytree = jax.tree.map(_pytree_leaf_desc, data)
  return fut


def nonblocking_load(directory: str | PathLike[str],
                     shardings: PyTreeT | None = None,
                     pytree: PyTreeT | None = None,
                     pickle_module: PickleModule | None = None,
                     tensorstore_specs: PyTreeT | None = None,
                     best_effort: bool = False) -> SerializationFuture:

  fut = SerializationFuture(async_load, directory, shardings, pytree,
                            pickle_module, tensorstore_specs, best_effort)
  # if user provided a pytree, then we'll use this
  # TODO(rdyro): technically, the user is expected to provide a pytree of UUIDs
  if pytree is None:
    pytree = load_pytree(directory, pickle_module, best_effort)
  #fut.pytree = _construct_shape_dtype_pytree_from_leaf_ids(pytree)
  fut.pytree = jax.tree.map(_leaf_desc_to_leaf, pytree)
  return fut
