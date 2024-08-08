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

import collections
from os import PathLike
from typing import Any, Sequence, TypeVar
from uuid import uuid4, UUID
import json
import asyncio
import threading
import shutil
import pickle
import functools
import time
import logging
import importlib
import itertools

import jax
from jax.tree_util import PyTreeDef, default_registry, treedef_is_leaf
from jax import tree
from jax._src.util import safe_zip
from jax._src import distributed
from jax.experimental.multihost_utils import broadcast_one_to_all
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
_TREE_REPR_KEY = "__jax_tree_repr"
_LEAF_IDS_KEY = "__jax_leaf_ids"
_TYPE_ID_LEAF_DELIMITER = " -> "
_USE_OCDBT = True
_SYNC_KEY_TIMEOUT_SEC = 300

__all__ = ["save", "load", "load_pytree", "async_save", 
           "async_load", "async_load_pytree"]

PyTreeT = Any
PickleModule = TypeVar("PickleModule")

def _get_unique_sync_key() -> str | None:
  """Generate a thread-local key for ensuring all host finish (de)serializing"""
  if jax.process_count() == 1:
    return None
  # broadcast a thread-local unique barrier name
  sync_key_id = UUID(bytes=np.array(broadcast_one_to_all(
    np.frombuffer(uuid4().bytes, dtype=np.int32))).tobytes())
  sync_key = f"jax_sync_key_{str(sync_key_id)}"
  return sync_key


def _sync_on_key(key: str | None) -> None:
  if key is None:
    return
  assert jax.process_count() > 1, (
    "You are attempting to wait for other hosts, but there is only 1 host"
  )
  distributed.global_state.client.wait_at_barrier(key, 
                                                  _SYNC_KEY_TIMEOUT_SEC * 1000)

def _is_array_like(x: Any):
  return isinstance(x, (jax.Array, np.ndarray))
  
def _get_suffix(x: Any):
  if _is_array_like(x):
    return _TENSORSTORE_SUFFIX
  elif isinstance(x, bytes):
    return ".bin"
  else:
    return ".json"

def _obj_serialize(x: Any, path: PathLike, 
                   pickle_module: PickleModule | None = None) -> str:
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
      if not pickle_module:
        raise ValueError(
          "An object cannot be serialized using JSON, your pytree contains a"
          f" non-serializable object: {x}. Consider using"
          " `pickle_module=pickle`")
      else:
        Path(path).write_bytes(pickle_module.dumps(x))
        return ".pickle"

def _obj_deserialize(path: PathLike, pickle_module: PickleModule | None = None, 
                     best_effort: bool = False):
  """Deserialization method for NON-array objects."""
  path = Path(path)
  suffix = path.suffix
  if _is_array_like(_TENSORSTORE_SUFFIX):
    raise ValueError
  elif suffix == ".bin":
    return path.read_bytes()
  elif suffix == ".json":
    return json.loads(path.read_text())
  elif suffix == ".pickle":
    if not pickle_module and not best_effort:
      raise ValueError(f"Pickle file encountered, but {pickle_module=}")
      return path.read_bytes()
    raw_bytes = path.read_bytes()
    if not pickle_module and best_effort:
      return raw_bytes
    try:
      return pickle_module.loads(raw_bytes)
    except (pickle.PickleError, AttributeError):
      raise ValueError(f"We couldn't deserialize obj at `{path.name}` using"
                       f" {pickle_module=}")
      logging.warning("We couldn't deserialize %s, we'll return underlying" 
                      " bytes instead", path.stem)
      return path.read_bytes()
  else:
    raise ValueError(f"Suffix `{suffix}` deserialization is not supported.")
    
def _leaf_to_type_desc(leaf) -> str:
  if leaf is None:
    return "null"
  elif isinstance(leaf, (np.ndarray, jax.Array)):
    return f"{leaf.dtype.name}[{', '.join(map(str, leaf.shape))}]"
  else:
    return type(leaf).__name__
    
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

async def _read_node_store(path: PathLike):
  """Read all binary files into the node_store - where node data lives."""
  path = Path(path)
  if not path.exists() or not path.is_dir():
    return {}
  paths = [path for path in Path(path).iterdir() 
           if path.is_file() and path.suffix == ".pickle"]
  contents = await asyncio.gather(*[asyncio.to_thread(
    lambda p: p.read_bytes(), path) for path in paths])
  return {path.stem: data for (path, data) in safe_zip(paths, contents)}


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
  return any(str(path).startswith(prefix[:-1]) 
             for prefix in _REMOTE_URL_PREFIXES)

async def async_save(data: PyTreeT, directory: str | PathLike, 
                     overwrite: bool = True, 
                     pickle_module: PickleModule | None = None) -> None:
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
  data_flat, pytreedef = tree.flatten(data)
  serialize_pytree = PermissivePyTreeSerialization.serialize_pytree

  # overwrite or error
  root = Path(directory)
  if ((root / _PYTREEDEF_FILE).exists() 
      or (root.exists() and len(list(root.iterdir())) > 0)):
    if overwrite:
      # check that we're only deleting things that come from JAX
      # refuse to rm directories containing additional entries
      paths_present = list(Path(directory).iterdir())
      extra_member_paths = [path for path in paths_present if path.name not in 
                            (_PYTREEDEF_FILE, _LEAF_DATA_DIR, _NODE_DATA_DIR)]
      assert len(extra_member_paths) == 0, (
        "Refusing to remove a directory that is not a previous checkpoint."
        f" Unrecognized paths: {extra_member_paths}. Remove them manually if"
        f" you're sure you want to use {root} as the checkpoint directory.")
      if _is_remote_path(directory):
        Path(directory).rmtree()
      else:
        shutil.rmtree(Path(directory))
    else:
      raise NotImplementedError("Another checkpoint already exists at path:"
                                f" `{root}`, but you specified `{overwrite=}`")

  if not _is_remote_path(directory):
    root = root.resolve()
    root.mkdir(exist_ok=True) # do not make parents, that's too much
    assert root.exists() and root.is_dir()

  pytree_repr, leaf_ids, node_data_store = serialize_pytree(pytreedef, 
                                                            pickle_module)
  leaf_ids_flat = tree.flatten(leaf_ids)[0]
  # optionally inscribe types into leaf ids
  leaf_ids_type_map = {leaf_id: _leaf_to_type_desc(leaf) for (leaf_id, leaf) 
                       in safe_zip(leaf_ids_flat, data_flat)}
  # in-place
  _inscribe_leaf_types(pytree_repr[_TREE_REPR_KEY], leaf_ids_type_map) 
  pytree_repr[_LEAF_IDS_KEY] = [
    _join_leaf_type_and_id(leaf_ids_type_map[leaf_id], leaf_id) 
    for leaf_id in leaf_ids_flat]

  serialize_futures = []

  # start serialization ##################################

  # 0. serialize the pytree
  if not _is_remote_path(root) or jax.process_index() == 0:
    serialize_futures.append(asyncio.to_thread(
      lambda: (root / _PYTREEDEF_FILE).write_text(
        json.dumps(pytree_repr, indent=2))))
    paths = [(root / _LEAF_DATA_DIR / f"{leaf_id}{_get_suffix(data)}") 
             for leaf_id, data in zip(leaf_ids_flat, data_flat)]


  # 1. serialize JSON serializable leafs
  obj_serialize_futures = []
  if not _is_remote_path(root) or jax.process_index() == 0:
    obj_flat = [data for path, data in zip(paths, data_flat) 
                if path.suffix != _TENSORSTORE_SUFFIX]
    obj_paths = [path for path in paths if path.suffix != _TENSORSTORE_SUFFIX]
    for data, path in zip(obj_flat, obj_paths):
      # each host has to read all binary files
      obj_serialize_futures.append(asyncio.to_thread(
        _obj_serialize, data, path, pickle_module=pickle_module))

  # 2. serialize arrays
  arr_flat = [data for path, data in zip(paths, data_flat) 
              if path.suffix == _TENSORSTORE_SUFFIX]
  arr_paths = [path for path in paths if path.suffix == _TENSORSTORE_SUFFIX]
  ts_specs = [get_tensorstore_spec(path, ocdbt=_USE_OCDBT) 
              for path in arr_paths]
  # primary host set to host 0 or None (all hosts write everything)
  primary_host = None if not _is_remote_path(directory) else jax.process_index()
  serialize_futures.extend([serialization.async_serialize(
    arr, ts_spec, primary_host=primary_host) 
    for (arr, ts_spec) in zip(arr_flat, ts_specs)])

  # 3. serialize node data if permissive
  if pickle_module is not None and (not _is_remote_path(root) or 
                                    jax.process_index() == 0):
    for node_data_key, node_data_bytes in node_data_store.items():
      path = root / _NODE_DATA_DIR / f"{node_data_key}.pickle"
      serialize_futures.append(asyncio.to_thread(_obj_serialize, 
                                                 node_data_bytes, path, 
                                                 pickle_module=pickle_module))
                      
  # perhaps rename some objects if pickle is provided and they turned out 
  # not to be JSON serializable (so we used pickle)
  _, obj_serialize_results = await asyncio.gather(
    asyncio.gather(*serialize_futures), asyncio.gather(*obj_serialize_futures))
  rename_futures = []
  if pickle_module is not None: # object suffixes might have changed
    for path, obj_actual_suffix in zip(obj_paths, obj_serialize_results):
      if path.suffix != obj_actual_suffix:
        rename_futures.append(
          asyncio.to_thread(lambda p, new_suffix: 
                            p.rename(p.with_suffix(new_suffix)), 
                            path, obj_actual_suffix))
  await asyncio.gather(*rename_futures)
  _sync_on_key(sync_key) # we are done with all async ops here, we can block


async def async_load(directory: str | PathLike, 
                     shardings: PyTreeT | None = None, 
                     pytree: PyTreeT | None = None,
                     pickle_module: PickleModule | None = None,
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

  data_paths = ([Path(path) for path in (root / _LEAF_DATA_DIR).iterdir()] 
                if (root / _LEAF_DATA_DIR).exists() else [])
  
  # deserialize in 3 stages

  # 1. deserialize PyTree (if permissive inserting node_data)
  if pytree is None:
    pytreedef = await async_load_pytree(directory, pickle_module, best_effort)
    leaf_ids, pytreedef = tree.flatten(pytreedef)
  else:
    leaf_ids, pytreedef = tree.flatten(pytree)
  leaf_ids = [leaf_id.split(_TYPE_ID_LEAF_DELIMITER)[1]
              if _TYPE_ID_LEAF_DELIMITER in leaf_id else leaf_id 
              for leaf_id in leaf_ids]
  leaf_ids_set = set(leaf_ids)
  missing_leaf_ids = [leaf_id for leaf_id in leaf_ids 
                      if leaf_id not in set(path.stem for path in data_paths)]
  if len(missing_leaf_ids) > 0:
    raise ValueError(
      f"Values {missing_leaf_ids} are missing from thecheckpoint directory.")
  
  # 2. deserialize non-array objects
  obj_paths = [path for path in data_paths 
               if path.suffix != _TENSORSTORE_SUFFIX]
  objs = {path.stem: asyncio.to_thread(
    _obj_deserialize, path, pickle_module, best_effort) for path in obj_paths 
    if path.stem in leaf_ids_set}

  # 3. deserialize array objects
  arr_paths = [path for path in data_paths 
               if path.suffix == _TENSORSTORE_SUFFIX]
  # missing sharding assumes we want to deserialize on default device
  if shardings is None:
    device = jax.devices()[0] # default device
    shardings = [SingleDeviceSharding(device) for _ in arr_paths]
  else:
    shardings = tree.flatten(shardings)[0]

  ts_specs = [get_tensorstore_spec(path, ocdbt=_USE_OCDBT)
              for path in arr_paths]
  arrs = {path.stem: serialization.async_deserialize(sharding, ts_spec) 
              for (sharding, ts_spec, path) 
              in zip(shardings, ts_specs, arr_paths)
              if path.stem in leaf_ids_set}
  arr_and_objs = objs | arrs # pythonic join dictionaries 
  filled_values = await asyncio.gather(*[arr_and_objs[leaf_id] 
                                         for leaf_id in leaf_ids])
  _sync_on_key(sync_key) # we are done with all async ops here, we can block
  return tree.unflatten(pytreedef, filled_values)

  
async def async_load_pytree(directory: str | PathLike, 
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
  raw_tree = json.loads((root / _PYTREEDEF_FILE).read_text())
  
  # detect if the written tree is permissive or strict
  if (isinstance(raw_tree, dict) and len(raw_tree) == 2 
      and all(key for key in raw_tree for key in 
              [_TREE_REPR_KEY, _LEAF_IDS_KEY])):
    # the checkpoint is permissive, ostensibly it needs a pickler to load binary
    # data, but the user might have used permissive mode to save 
    # a strict / clean checkpoint, error lazily
    if pickle_module is not None:
      node_data_store = await _read_node_store(root / _NODE_DATA_DIR)
    elif (not best_effort and (root / _NODE_DATA_DIR).exists() 
          and len(list((root / _NODE_DATA_DIR).iterdir())) > 0):
      raise ValueError(
        f"{_NODE_DATA_DIR} is not empty, but pickle_module is not provided and"
        f" `{best_effort=}`. We cannot proceed.")
    else:
      node_data_store = dict()
    return PermissivePyTreeSerialization.deserialize_pytree(
      raw_tree, node_data_store, pickle_module, best_effort)
  else:
    # the checkpoint is strict / not permissive
    return StrictPyTreeSerialization.deserialize_pytree(raw_tree)


def _maybe_run_async_sync(name, async_fn):
  """Run async routine synchronously irrespective of the current environment."""

  def wrapped_fn(*args, **kw):
    retval, exception = [None], [None]

    def _run_in_thread():
      async def _async_thread_run():
        ret, exc = None, None
        try:
          ret = await async_fn(*args, **kw)
        except Exception as e:
          exc = e
        return ret, exc

      retval[0], exception[0] = asyncio.run(_async_thread_run())

    t = threading.Thread(target=_run_in_thread)
    t.start()
    t.join()
    if exception[0] is not None:
      raise exception[0]
    return retval[0]
    
  functools.update_wrapper(wrapper=wrapped_fn, wrapped=async_fn)
  wrapped_fn.__name__ = name
  wrapped_fn.__qualname__ = name
  return wrapped_fn

################################################################################

load_pytree = _maybe_run_async_sync("load_pytree", async_load_pytree)
save = _maybe_run_async_sync("save", async_save)
load = _maybe_run_async_sync("load", async_load)
  
################################################################################


def _cls2typerepr(cls):
  return f"{cls.__module__}.{cls.__name__}"

# extended node types map for pickle-free nodes
# we limit ourselves here to an enumerate list for safety
# users who have additional needs can always fall-back to pickle
_EXTENDED_NODE_TYPES_MAP = {
  "builtins.dict": dict,
  "builtins.list": list,
  "builtins.tuple": tuple,
  "builtins.set": set,
  _cls2typerepr(collections.OrderedDict): collections.OrderedDict,
  "flax.core.frozen_dict.FrozenDict": "flax.core.frozen_dict.FrozenDict",
}
  
def _encode_node_data(node_data, node_data_store: dict[str, Any], 
                      pickle_module: PickleModule | None):
  if node_data is None:
    return "leaf", None
  elif _cls2typerepr(node_data[0]) in _EXTENDED_NODE_TYPES_MAP:
    return (_cls2typerepr(node_data[0]), node_data[1])
  else:
    # the fallback case
    node_data_key = str(uuid4())
    if pickle_module is None:
      raise ValueError(
        f"Node data `{node_data}` is not serializable without a pickle_module.")
    # we pickle node_data which includes **both** the class and data
    node_data_store[node_data_key] = pickle_module.dumps(node_data)
    return (_cls2typerepr(node_data[0]), node_data_key)
    
def _decode_extended_type(typerepr: str, node_data: Any, best_effort: bool):
  """Given a type name like `flax.core.frozen_dict` try to return the class."""
  module_name, cls_name = typerepr.rsplit(".", 1)
  try:
    missing_attribute = False
    module = importlib.import_module(module_name)
    if not hasattr(module, cls_name):
      missing_attribute = True
      raise ImportError
    node_cls = getattr(module, cls_name)
  except ImportError:
    missing_mod_msg = f"Could not find module `{module_name}`."
    missing_cls_msg = (f"Coud not find class `{cls_name}` in module"
                        f" `{module_name}`.")
    if best_effort:
      msg = f" Falling back to a list of children since {best_effort=}."
      logging.warning((missing_mod_msg if missing_attribute 
                        else missing_cls_msg) + msg)
      node_cls, node_data = (list, None)
    else:
      raise ImportError(missing_mod_msg if missing_attribute 
                        else missing_cls_msg)
  return node_cls, node_data

def _decode_node_data(node_type, node_data, node_data_store: dict[str, Any], 
                      pickle_module: PickleModule, best_effort: bool):
  if node_type is None or node_type == "leaf":
    return None
  elif node_type in _EXTENDED_NODE_TYPES_MAP:
    node_class_or_str = _EXTENDED_NODE_TYPES_MAP[node_type]
    if isinstance(node_class_or_str, type):
      node_cls = node_class_or_str
    else:
      node_cls, node_data = _decode_extended_type(node_class_or_str, node_data, 
                                                  best_effort)
    return (node_cls, node_data)
  else:
    node_data_ref = node_data
    if not best_effort:
      assert node_data_ref in node_data_store, (
        f"Node reference `{node_data_ref}` missing from node data store.")
    if node_data_ref in node_data_store:
      pickle_error = None
      try:
        assert pickle_module is not None
        return pickle_module.loads(node_data_store[node_data_ref])
      except (pickle.UnpicklingError, AttributeError) as e:
        pickle_error = e
    # fallback
    if best_effort:
      logging.warning(f"Unrecognized data type `{node_type}` we'll do"
                      " our best and just return a list of children")
      return (list, None)
    else:
      raise pickle_error


def _serialize_pytree_helper(node, node_data_store: dict[str, Any], 
                             pickle_module: PickleModule | None):
  node_repr = dict()
  node_repr["node_type"], node_repr["node_data_ref"] = _encode_node_data(
    node.node_data(), node_data_store, pickle_module)
  # this encodes if-and-only-if using not and xor
  assert not ((len(node.children()) == 0) ^ treedef_is_leaf(node))

  leaf_ids = None
  if not treedef_is_leaf(node):
    children, leaf_ids = zip(*[
      _serialize_pytree_helper(child, node_data_store=node_data_store, 
                               pickle_module=pickle_module) 
      for child in node.children()])
    node_repr["children"] = children
  else:
    node_repr["children"] = []
    assert node.num_leaves in (0, 1), (
      "We only support 1 or 0 leaves (?) is this a leaf")
    leaf_id = str(uuid4()) if node.num_leaves == 1 else None
    node_repr["leaf_id"] = leaf_id
    leaf_ids = node_repr["leaf_id"]
  return node_repr, leaf_ids

def _deserialize_pytree_helper(node, node_data_store: dict[str, Any],
                               pickle_module: PickleModule | None, 
                               best_effort: bool):
    assert ("node_type" in node and "node_data_ref" in node 
            and "children" in node)
    pytree_children = [_deserialize_pytree_helper(
      child, node_data_store, pickle_module, best_effort) 
      for child in node["children"]]
    node_data = _decode_node_data(node["node_type"], node["node_data_ref"], 
                                  node_data_store, pickle_module, best_effort)
    pt = PyTreeDef.make_from_node_data_and_children(default_registry, node_data,
                                                    tuple(pytree_children))
    return pt

# serialize and deserialize pytree methods namespaces: permissive and strict
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
  def serialize_pytree(node, pickle_module: PickleModule | None = None):
    node_data_store = dict()
    tree_repr, leaf_ids = _serialize_pytree_helper(node, node_data_store, 
                                                   pickle_module)
    node_repr = dict()
    node_repr[_TREE_REPR_KEY] = tree_repr
    node_repr[_LEAF_IDS_KEY] = tree.flatten(leaf_ids)[0]
    return node_repr, leaf_ids, node_data_store

  @staticmethod
  def deserialize_pytree(rawtree: dict[str, Any], 
                         node_data_store: dict[str, Any] | None = None,
                         pickle_module: PickleModule | None = None,
                         best_effort: bool = False):
    node_data_store = node_data_store or dict()
    pt = _deserialize_pytree_helper(rawtree[_TREE_REPR_KEY], node_data_store, 
                                    pickle_module, best_effort)
    leaf_ids = rawtree[_LEAF_IDS_KEY]
    return tree.unflatten(pt, leaf_ids)
    
class StrictPyTreeSerialization(AbstractPyTreeSerialization):
  @staticmethod
  def serialize_pytree(node, pickle_module: PickleModule | None = None):
    del pickle_module
    leaf_ids = [str(uuid4()) for _ in range(node.num_leaves)]
    _ = PyTreeDef.serialize_using_proto(node) # this raises a useful error
    # alternatively: assert _is_pytree_strict_serializable(node)
    node_repr = tree.unflatten(node, leaf_ids)
    return node_repr, leaf_ids, dict()

  @staticmethod
  def deserialize_pytree(rawtree: dict[str, Any], 
                         node_data_store: dict[str, Any] | None = None,
                         pickle_module: PickleModule | None = None,
                         best_effort: bool = False):
    del node_data_store
    del pickle_module
    del best_effort
    return rawtree