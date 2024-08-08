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

import jax
from jax.tree_util import PyTreeDef, default_registry
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

def serialize_pytreedef(tree: PyTreeDef) -> bytes:
  # check that the tree deserializes with the default registry
  try:
    pytreedef_data = PyTreeDef.serialize_using_proto(tree)
  except TypeError:
    raise NotImplementedError("Attempting to serialize a PyTreeDef that does"
                              " has custom types. This is not currently"
                              " supported.")
  return pytreedef_data

def deserialize_pytreedef(tree_bytes: bytes, 
                          registry = default_registry) -> PyTreeDef:
  return PyTreeDef.deserialize_using_proto(registry, tree_bytes)
  

async def async_save(data: Any, directory: str | PathLike, 
                     overwrite: bool = True) -> None:
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

  # serialize non-array objects, avoid the zip(*[]) for clarity 
  obj_flat = [data for path, data in zip(paths, data_flat) 
              if path.suffix != _TENSORSTORE_SUFFIX]
  obj_paths = [path for path in paths if path.suffix != _TENSORSTORE_SUFFIX]

  serialize_futures = []

  for data, path in zip(obj_flat, obj_paths):
    serialize_futures.append(asyncio.to_thread(_obj_serialize, data, path))

  # serialize arrays, avoid the zip(*[]) for clarity 
  arr_flat = [data for path, data in zip(paths, data_flat) 
              if path.suffix == _TENSORSTORE_SUFFIX] 
  arr_paths = [path for path in paths if path.suffix == _TENSORSTORE_SUFFIX]
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
  arr_objs = objs | arrs
  filled_values = await asyncio.gather(*[arr_objs[value] for value in values])
  return tree.unflatten(pytreedef, filled_values)

  
def load_pytree(data: Any, directory: str | PathLike) -> None:
  def _async_save():
    asyncio.run(async_save(data=data, directory=directory))

  thread = threading.Thread(target=_async_save)
  thread.start()
  thread.join()

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