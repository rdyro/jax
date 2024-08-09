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

__all__ = ["save", "load", "async_save", "async_load"]


def _is_array_like(x: Any):
  return isinstance(x, (jax.Array, np.ndarray))
  
def _get_suffix(x: Any):
  if _is_array_like(x):
    return _TENSORSTORE_SUFFIX
  elif isinstance(x, bytes):
    return ".bin"
  else:
    return ".json"

def _obj_serialize(x: Any, path: PathLike):
  """Serialization method for NOT-array objects."""
  if _is_array_like(x):
    raise ValueError
  elif isinstance(x, bytes):
    Path(path).write_bytes(x)
  else:
    Path(path).write_text(json.dumps(x))

def _obj_deserialize(path: PathLike):
  """Deserialization method for NOT-array objects."""
  path = Path(path)
  suffix = path.suffix
  if _is_array_like(_TENSORSTORE_SUFFIX):
    raise ValueError
  elif suffix == ".bin":
    return path.read_bytes()
  elif suffix == ".json":
    return json.loads(path.read_text())
  else:
    raise ValueError(f"Suffix `{suffix}` deserialization is not supported.")

def _is_pytree_serializable(tree: PyTreeDef) -> bool:
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
                     overwrite: bool = True, custom_types: bool = False) -> None:
  if custom_types:
    raise NotImplementedError("We don't currently support saving pytrees with"
                              " custom types.")
  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`."
  )
  data_flat, pytreedef = tree.flatten(data)
  assert _is_pytree_serializable(pytreedef)

  root = Path(directory)
  if (root / _PYTREEDEF_FILE).exists() or len(list(root.glob("*"))) > 0:
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

  field_names = [str(uuid4()) for _ in data_flat]
  (root / _PYTREEDEF_FILE).write_text(
    json.dumps(tree.unflatten(pytreedef, field_names), indent=2))
  paths = [root / f"{field_name}{_get_suffix(data)}"
                   for field_name, data in zip(field_names, data_flat)]

  obj_flat, obj_paths = zip(*[(data, path) 
                              for path, data in zip(paths, data_flat) 
                              if path.suffix != _TENSORSTORE_SUFFIX])
  serialize_futures = []

  for data, path in zip(obj_flat, obj_paths):
    serialize_futures.append(asyncio.to_thread(_obj_serialize, data, path))

  arr_flat, arr_paths = zip(*[(data, path) for path, data 
                              in zip(paths, data_flat) 
                              if path.suffix == _TENSORSTORE_SUFFIX])
  ts_specs = [get_tensorstore_spec(path) for path in arr_paths]
  serialize_futures.extend([serialization.async_serialize(arr, ts_spec) 
                            for (arr, ts_spec) in zip(arr_flat, ts_specs)])
  await asyncio.gather(*serialize_futures)

async def async_load(directory: str | PathLike, 
                     shardings: Sequence | None = None, 
                     pytree: Any | None = None) -> Any:
  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`.")
  root = Path(directory)
  paths = [Path(path) for path in root.glob("*") if 
           Path(path).name != _PYTREEDEF_FILE]
  if pytree is None:
    pytree = json.loads((root / _PYTREEDEF_FILE).read_text())
  values, pytreedef = tree.flatten(pytree)
  missing_paths = [value for value in values 
                   if value not in set(path.stem for path in paths)]
  if len(missing_paths) > 0:
    raise ValueError(f"Values {missing_paths} are missing from the checkpoint"
                     " directory.")
  
  obj_paths = [path for path in paths if path.suffix != _TENSORSTORE_SUFFIX]
  objs = {path.stem: asyncio.to_thread(_obj_deserialize, path) 
          for path in obj_paths}
  arr_paths = [path for path in paths if path.suffix == _TENSORSTORE_SUFFIX]

  # missing sharding assumes we want to deserialize on default device
  if shardings is None:
    device = jax.devices()[0] # default device
    shardings = [SingleDeviceSharding(device) for _ in arr_paths]
  else:
    shardings = tree.flatten(shardings)[0]

  ts_specs = [get_tensorstore_spec(path) for path in arr_paths]
  arrs = {path.stem: serialization.async_deserialize(sharding, ts_spec) 
              for (sharding, ts_spec, path) 
              in zip(shardings, ts_specs, arr_paths)}
  arr_objs = objs | arrs # pythonic join dictionaries 
  filled_values = await asyncio.gather(*[arr_objs[value] for value in values])
  return tree.unflatten(pytreedef, filled_values)

  
def load_pytree(directory: str | PathLike) -> None:
  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`.")
  root = Path(directory)
  return json.loads((root / _PYTREEDEF_FILE).read_text())

def save(data: Any, directory: str | PathLike) -> None:
  def _async_save():
    asyncio.run(async_save(data=data, directory=directory))

  thread = threading.Thread(target=_async_save)
  thread.start()
  thread.join()

def load(directory: str | PathLike, shardings: Sequence | None = None,
         pytree: Any | None = None) -> Any:
  result = []
  def _async_load():
    result.append(asyncio.run(async_load(directory=directory, pytree=pytree,
                                         shardings=shardings)))
  thread = threading.Thread(target=_async_load)
  thread.start()
  thread.join()
  return result[0]
  
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
    node_data_bytes = pickle.dumps(node_data)
    node_data_key = str(uuid4())
    node_data_store[node_data_key] = node_data_bytes
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
  node_data_store = node_data_store or dict()
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
    return node_repr, leaf_ids, node_data_store

  @staticmethod
  def deserialize_pytree(node: dict[str, Any], 
                        node_data_store: dict[str, Any] | None = None):
    if node_data_store is None:
      node_data_store = dict()
    assert "encoded_node_data" in node and "children" in node
    pytree_children = [PermissivePyTreeSerialization.deserialize_pytree(child) 
                       for child in node["children"]]
    node_data = _decode_node_data(node["encoded_node_data"], node_data_store)
    pt = PyTreeDef.make_from_node_data_and_children(default_registry, node_data,
                                                    pytree_children)
    return pt

class StrictPyTreeSerialization(AbstractPyTreeSerialization):
  @staticmethod
  def serialize_pytree(node):
    leaf_ids = [str(uuid4()) for _ in range(node.num_leaves)]
    assert _is_pytree_serializable(node)
    node_repr = tree.unflatten(node, leaf_ids)
    node_repr["permissive"] = False
    return node_repr, leaf_ids, dict()

  @staticmethod
  def deserialize_pytree(node: dict[str, Any], 
                         node_data_store: dict[str, Any] | None = None):
    return tree.structure(node)