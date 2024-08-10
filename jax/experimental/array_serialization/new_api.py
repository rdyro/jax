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
from typing import Any, Sequence
from uuid import uuid4
import json
import asyncio
import threading
import shutil
import pickle
import logging
from dataclasses import dataclass

import jax
from jax.tree_util import PyTreeDef, default_registry, treedef_is_leaf
from jax import tree
from jax.experimental.array_serialization import serialization
from jax.experimental.array_serialization.serialization import get_tensorstore_spec
from jax.experimental.array_serialization.serialization import _REMOTE_URL_PREFIXES
from jax.sharding import SingleDeviceSharding
from jax._src.path import epath_installed, Path
import numpy as np

_PYTREEDEF_FILE = "pytreedef.json"
_TENSORSTORE_SUFFIX = ".tensorstore"
_LEAF_DATA_DIR = "leaf_data"
_NODE_DATA_DIR = "node_data"
_PERMISSIVE_TREE_METADATA = ["permissive", "flat_leaf_ids"]

__all__ = ["save", "load", "load_pytree", "async_save", 
           "async_load", "async_load_pytree"]



def _is_array_like(x: Any):
  return isinstance(x, (jax.Array, np.ndarray))
  
def _get_suffix(x: Any):
  if _is_array_like(x):
    return _TENSORSTORE_SUFFIX
  elif isinstance(x, bytes):
    return ".bin"
  else:
    return ".json"

def _obj_serialize(x: Any, path: PathLike, use_pickle: bool) -> str:
  """Serialization method for NOT-array objects."""
  if _is_array_like(x):
    raise ValueError
  elif isinstance(x, bytes):
    Path(path).parent.mkdir(exist_ok=True)
    Path(path).write_bytes(x)
    return ".bin"
  elif Path(path).suffix == ".pickle":
    Path(path).parent.mkdir(exist_ok=True)
    Path(path).write_bytes(pickle.dumps(x))
  else: # attempt to write JSON - the default
    Path(path).parent.mkdir(exist_ok=True)
    try:
      Path(path).write_text(json.dumps(x))
      return ".json"
    except TypeError: # object is not JSON serializable
      if not use_pickle:
        raise ValueError(
          "An object cannot be serialized using JSON, your pytree contains a"
          f" non-serializable object: {x}. Consider using `permissive=True`")
      else:
        Path(path).write_bytes(pickle.dumps(x))
        return ".pickle"

def _obj_deserialize(path: PathLike, use_pickle: bool):
  """Deserialization method for NOT-array objects."""
  path = Path(path)
  suffix = path.suffix
  if _is_array_like(_TENSORSTORE_SUFFIX):
    raise ValueError
  elif suffix == ".bin":
    return path.read_bytes()
  elif suffix == ".json":
    return json.loads(path.read_text())
  elif suffix == ".pickle":
    if not use_pickle:
      logging.warning(f"Pickle file encountered, but `{use_pickle=}`," 
                      " we'll return raw bytes")
      return path.read_bytes()
    try:
      return pickle.loads(path.read_bytes())
    except (pickle.PickleError, AttributeError):
      logging.warning("We couldn't deserialize %s, we'll return underlying" 
                      " bytes instead", path.stem)
      return path.read_bytes()
  else:
    raise ValueError(f"Suffix `{suffix}` deserialization is not supported.")


async def _read_node_store(path: PathLike):
  """Unpickle all pickle files in the node_store - where node data lives."""
  path = Path(path)
  if not path.exists() or not path.is_dir():
    return {}
  paths = [path for path in Path(path).iterdir() 
           if path.is_file() and path.suffix == ".pickle"]
  contents = await asyncio.gather(*[asyncio.to_thread(
    lambda p: p.read_bytes(), path) for path in paths])
  return {path.stem: data for (path, data) in zip(paths, contents)}


def _is_pytree_strict_serializable(tree: PyTreeDef) -> bool:
  """Check if pytree is free of custom nodes."""
  try:
    _ = PyTreeDef.serialize_using_proto(tree)
    return True
  except TypeError:
    return False
    
def _is_remote_path(path: str | PathLike):
  # we check whether a path is remote by checking the prefix
  # we need to truncate e.g., gs:// to gs:/ because pathlib.Path collapses //
  return any(str(path).startswith(prefix[:-1]) for prefix in 
             _REMOTE_URL_PREFIXES)

async def async_save(data: Any, directory: str | PathLike, 
                     overwrite: bool = True, permissive: bool = False) -> None:
  """Saves the given data structure to the provided directory path.

  This function provides functionality to serialize and save a data structure
  comprising JAX arrays, NumPy arrays, Python objects, etc., along with its
  structure to a given directory. It leverages `PyTree` for flattening and
  reconstructing the data structure.

  If `permissive` is enabled, we use pickle to serialize unknown types.

  Args:
    data: The data structure to be saved. Arbitrary composition of JAX arrays, 
      NumPy arrays, and Python objects, including nested structures.
    directory: The directory path where the data will be saved. A local path or 
      a remote URL (e.g., gs://, s3://). For remote URLs, `etils` is required.
    overwrite: If True, any existing directory with the same name will be
      overwritten.
    permissive: If True, uses a permissive serialization strategy that uses
      cloudpickle to serialize any unsupported types.
  Raises:
    AssertionError: If attempting to save to a remote path without the `etils`
      package installed.
    NotImplementedError: If `overwrite` is False and a checkpoint already
      exists at the provided directory.
  """

  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`."
  )
  data_flat, pytreedef = tree.flatten(data)
  if permissive:
    serialize_pytree = PermissivePyTreeSerialization.serialize_pytree
  else:
    serialize_pytree = StrictPyTreeSerialization.serialize_pytree

  root = Path(directory)
  if ((root / _PYTREEDEF_FILE).exists() 
      or (root.exists() and len(list(root.iterdir())) > 0)):
    if overwrite:
      if _is_remote_path(directory):
        Path(directory).rmtree()
      else:
        shutil.rmtree(Path(directory))
    else:
      raise NotImplementedError("Another checkpoint already exists at path:"
                                f" `{root}`, but you specified `overwrite ="
                                " False`")
  if not _is_remote_path(directory):
    root.mkdir(exist_ok=True)

  pytree_repr, leaf_ids, node_data_store = serialize_pytree(pytreedef)
  leaf_ids_flat = tree.flatten(leaf_ids)[0]
  (root / _PYTREEDEF_FILE).write_text(json.dumps(pytree_repr, indent=2))
  paths = [root / _LEAF_DATA_DIR / f"{leaf_id}{_get_suffix(data)}"
                   for leaf_id, data in zip(leaf_ids_flat, data_flat)]


  # start serialization ##################################

  # 1. serialize JSON serializable leafs
  obj_flat = [data for path, data in zip(paths, data_flat) 
              if path.suffix != _TENSORSTORE_SUFFIX]
  obj_paths = [path for path in paths if path.suffix != _TENSORSTORE_SUFFIX]
  obj_serialize_futures = []
  for data, path in zip(obj_flat, obj_paths):
    obj_serialize_futures.append(asyncio.to_thread(_obj_serialize, data, path, 
                                                   use_pickle=permissive))

  # 2. serialize arrays
  serialize_futures = []
  arr_flat = [data for path, data in zip(paths, data_flat) 
              if path.suffix == _TENSORSTORE_SUFFIX]
  arr_paths = [path for path in paths if path.suffix == _TENSORSTORE_SUFFIX]
  ts_specs = [get_tensorstore_spec(path) for path in arr_paths]
  serialize_futures.extend([serialization.async_serialize(arr, ts_spec) 
                            for (arr, ts_spec) in zip(arr_flat, ts_specs)])

  # 3. serialize node data if permissive
  if permissive:
    for node_data_key, node_data_bytes in node_data_store.items():
      path = root / _NODE_DATA_DIR / f"{node_data_key}.pickle"
      serialize_futures.append(asyncio.to_thread(_obj_serialize, 
                                                 node_data_bytes, path, 
                                                 use_pickle=True))
                      
  # perhaps rename some objects if permissive is enabled and they turned out 
  # not to be JSON serializable (so we used pickle)
  _, obj_serialize_results = await asyncio.gather(
    asyncio.gather(*serialize_futures), asyncio.gather(*obj_serialize_futures))
  rename_futures = []
  if permissive: # object suffixes might have changes
    for path, obj_actual_suffix in zip(obj_paths, obj_serialize_results):
      if path.suffix != obj_actual_suffix:
        rename_futures.append(
          asyncio.to_thread(lambda p, new_suffix: 
                            p.rename(p.with_suffix(new_suffix)), 
                            path, obj_actual_suffix))
  await asyncio.gather(*rename_futures)


async def async_load(directory: str | PathLike, 
                     shardings: Sequence | None = None, 
                     pytree: Any | None = None,
                     use_pickle: bool = True) -> Any:
  """Loads and reconstructs a data structure from a directory.

  Args:
    directory: Directory path where the data is stored.
    shardings: Sharding strategy for array objects. If None, defaults to
      single device sharding on the default device.
    pytree: Optional pre-populated PyTree for structure. If provided, must 
      specify a pytree with string object ids. Useful for partial reads.
    permissive: Whether to attempt to unpickle.
  Returns:
    Reconstructed data structure.
  Raises:
    AssertionError: If attempting to load from a remote path without etils
      installed.
    ValueError: If data for specific leaf IDs is missing in the directory.
  """
  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`.")
  root = Path(directory)
  data_paths = [Path(path) for path in (root / _LEAF_DATA_DIR).iterdir()]
  
  # deserialize in 3 stages

  # 1. deserialize PyTree (if permissive inserting node_data)
  if pytree is None:
    pytreedef, permissive = (
      await _async_load_pytree_and_check_permissive(directory, use_pickle))
    leaf_ids, pytreedef = tree.flatten(pytreedef)
  else:
    permissive = True
    leaf_ids, pytreedef = tree.flatten(pytree)
  leaf_ids_set = set(leaf_ids)
  missing_leaf_ids = [leaf_id for leaf_id in leaf_ids 
                   if leaf_id not in set(path.stem for path in data_paths)]
  if len(missing_leaf_ids) > 0:
    raise ValueError(
      f"Values {missing_leaf_ids} are missing from thecheckpoint directory.")
  
  # 2. deserialize non-array objects
  obj_paths = [path for path in data_paths 
               if path.suffix != _TENSORSTORE_SUFFIX]
  objs = {path.stem: asyncio.to_thread(_obj_deserialize, path, use_pickle) 
          for path in obj_paths if path.stem in leaf_ids_set}

  # 3. deserialize array objects
  arr_paths = [path for path in data_paths 
               if path.suffix == _TENSORSTORE_SUFFIX]
  # missing sharding assumes we want to deserialize on default device
  if shardings is None:
    device = jax.devices()[0] # default device
    shardings = [SingleDeviceSharding(device) for _ in arr_paths]
  else:
    shardings = tree.flatten(shardings)[0]

  ts_specs = [get_tensorstore_spec(path) for path in arr_paths]
  arrs = {path.stem: serialization.async_deserialize(sharding, ts_spec) 
              for (sharding, ts_spec, path) 
              in zip(shardings, ts_specs, arr_paths)
              if path.stem in leaf_ids_set}
  arr_and_objs = objs | arrs # pythonic join dictionaries 
  filled_values = await asyncio.gather(*[arr_and_objs[leaf_id] 
                                         for leaf_id in leaf_ids])
  return tree.unflatten(pytreedef, filled_values)

  
async def async_load_pytree(directory: str | PathLike, 
                            use_pickle: bool = True) -> Any:
  """Loads a pytree from the given directory.
  Args:
    directory: Directory path to load from.
  Returns:
    The loaded pytree.
  """
  return (await _async_load_pytree_and_check_permissive(directory, 
                                                        use_pickle))[0]

async def _async_load_pytree_and_check_permissive(
    directory: str | PathLike, use_pickle: bool = True) -> Any:
  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`.")
  root = Path(directory)
  raw_tree = json.loads((root / _PYTREEDEF_FILE).read_text())
  if raw_tree.get("permissive", False):
    if use_pickle:
      node_data_store = await _read_node_store(root / _NODE_DATA_DIR)
    else:
      node_data_store = dict()
    return (PermissivePyTreeSerialization.deserialize_pytree(
      raw_tree, node_data_store), True)
  else:
    # delete permissive tree metadata if present
    for key in [k for k in _PERMISSIVE_TREE_METADATA if k in raw_tree]: 
      del raw_tree[key]
    return StrictPyTreeSerialization.deserialize_pytree(raw_tree), False
    
def _run_async_in_thread_wrapper(fn):

  def wrapped_fn(*args, **kw):
    result = []

    def _async_run():
      result.append(asyncio.run(fn(*args, **kw)))

    thread = threading.Thread(target=_async_run)
    thread.start()
    thread.join()
    if len(result) == 0:
      raise ValueError(f"Underlying async method failed: `{fn}`")
    return result[0]

  return wrapped_fn
    
load_pytree = _run_async_in_thread_wrapper(async_load_pytree)
save = _run_async_in_thread_wrapper(async_save)
load = _run_async_in_thread_wrapper(async_load)
  
################################################################################

_BUILTINS_MAP = {
  "builtins.dict": dict,
  "builtins.list": list,
  "builtins.tuple": tuple,
  "builtins.set": set,
}
_BUILTINS_SET = set(_BUILTINS_MAP.values())

def _cls2typerepr(cls):
  return f"{cls.__module__}.{cls.__name__}"
  
def _encode_node_data(node_data, node_data_store: dict[str, Any]):
  if node_data is None:
    return None
  elif node_data[0] in _BUILTINS_SET:
    return (_cls2typerepr(node_data[0]), node_data[1])
  else:
    node_data_key = str(uuid4())
    node_data_store[node_data_key] = pickle.dumps(node_data)
    return (_cls2typerepr(node_data[0]), node_data_key)

def _decode_node_data(node_data_encoded, node_data_store: dict[str, Any]):
  if node_data_encoded is None:
    return None
  elif node_data_encoded[0] in _BUILTINS_MAP:
    return (_BUILTINS_MAP[node_data_encoded[0]], node_data_encoded[1])
  else:
    node_data_key = node_data_encoded[1]
    if node_data_key in node_data_store:
      try:
        return pickle.loads(node_data_store[node_data_key])
      except (pickle.UnpicklingError, AttributeError):
        pass
    # fallback
    logging.warning(f"Unrecognized data type {node_data_encoded[0]} we'll do"
                    " our best and just return a list of children")
    return (list, None)

def _serialize_pytree_helper(node, node_data_store: dict[str, Any]):
  node_repr = dict()
  node_repr["encoded_node_data"] = _encode_node_data(node.node_data(), 
                                                     node_data_store)
  # this encodes if-and-only-if using not and xor
  assert not ((len(node.children()) == 0) ^ treedef_is_leaf(node))

  leaf_ids = None
  if not treedef_is_leaf(node):
    children, leaf_ids = zip(*[
      _serialize_pytree_helper(child, node_data_store=node_data_store) 
      for child in node.children()])
    node_repr["children"] = children
  else:
    node_repr["children"] = []
    assert node.num_leaves in (0, 1), (
      "We only support 1 or 0 leaves (?) are a leaf")
    node_repr["leaf_id"] = str(uuid4()) if node.num_leaves == 1 else None
    leaf_ids = node_repr["leaf_id"]
  return node_repr, leaf_ids

def _deserialize_pytree_helper(node, node_data_store: dict[str, Any]):
    assert "encoded_node_data" in node and "children" in node
    pytree_children = [_deserialize_pytree_helper(
      child, node_data_store=node_data_store) for child in node["children"]]
    node_data = _decode_node_data(node["encoded_node_data"], node_data_store)
    pt = PyTreeDef.make_from_node_data_and_children(default_registry, node_data,
                                                    tuple(pytree_children))
    return pt

class AbstractPyTreeSerialization:
  @staticmethod
  def serialize_pytree(node):
    raise NotImplementedError

  @staticmethod
  def deserialize_pytree(node: dict[str, Any], 
                         node_data_store: dict[str, Any] | None = None):
    raise NotImplementedError

class PermissivePyTreeSerialization(AbstractPyTreeSerialization):
  @staticmethod
  def serialize_pytree(node):
    node_data_store = dict()
    node_repr, leaf_ids = _serialize_pytree_helper(node, node_data_store)
    node_repr["permissive"] = True
    node_repr["flat_leaf_ids"] = tree.flatten(leaf_ids)[0]
    return node_repr, leaf_ids, node_data_store

  @staticmethod
  def deserialize_pytree(rawtree: dict[str, Any], 
                         node_data_store: dict[str, Any] | None = None):
    node_data_store = node_data_store or dict()
    pt = _deserialize_pytree_helper(rawtree, node_data_store)
    leaf_ids = rawtree["flat_leaf_ids"]
    return tree.unflatten(pt, leaf_ids)
    
class StrictPyTreeSerialization(AbstractPyTreeSerialization):
  @staticmethod
  def serialize_pytree(node):
    leaf_ids = [str(uuid4()) for _ in range(node.num_leaves)]
    assert _is_pytree_strict_serializable(node)
    node_repr = tree.unflatten(node, leaf_ids)
    node_repr["permissive"] = False
    return node_repr, leaf_ids, dict()

  @staticmethod
  def deserialize_pytree(rawtree: dict[str, Any], 
                         node_data_store: dict[str, Any] | None = None):
    return rawtree