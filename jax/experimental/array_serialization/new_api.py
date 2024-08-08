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

import jax
from jax.tree_util import PyTreeDef, default_registry
from jax import tree
from jax.experimental.array_serialization import serialization
from jax.experimental.array_serialization.serialization import get_tensorstore_spec
from jax.sharding import SingleDeviceSharding
from etils import epath
import numpy as np

_PYTREEDEF_FILE = "pytreedef.json"

__all__ = ["save", "load", "async_save", "async_load"]


def _is_array_like(x: Any):
  return isinstance(x, (jax.Array, np.ndarray))

def _is_pytree_serializable(tree: PyTreeDef) -> bool:
  try:
    _ = PyTreeDef.serialize_using_proto(tree)
    return True
  except TypeError:
    return False

def serialize_pytreedef(tree: PyTreeDef) -> bytes:
  # check that the tree deserializes with the default registry
  try:
    #deserialize_pytreedef(pytreedef_data)
    pytreedef_data = PyTreeDef.serialize_using_proto(tree)
  except TypeError:
    raise NotImplementedError("Attempting to serialize a PyTreeDef that does"
                              " has custom types. This is not currently"
                              " supported.")
  return pytreedef_data

def deserialize_pytreedef(tree_bytes: bytes, 
                          registry = default_registry) -> PyTreeDef:
  return PyTreeDef.deserialize_using_proto(registry, tree_bytes)
  

async def async_save(data: Any, directory: str | PathLike) -> None:
  data_flat, pytreedef = tree.flatten(data)
  assert _is_pytree_serializable(pytreedef)

  root = epath.Path(directory)
  if (root / _PYTREEDEF_FILE).exists() or len(list(root.glob("*"))) > 0:
    raise NotImplementedError("Another checkpoint already exists at path:"
                              f" `{root}`.")
  root.mkdir(exist_ok=True)

  field_names = [str(uuid4()) for _ in data_flat]
  (root / _PYTREEDEF_FILE).write_text(
    json.dumps(tree.unflatten(pytreedef, field_names), indent=2))
  paths = [root / (field_name if _is_array_like(data) else f"{field_name}.json")
                   for field_name, data in zip(field_names, data_flat)]

  # serialize non-array objects, avoid the zip(*[]) for clarity 
  obj_flat = [data for path, data in zip(paths, data_flat) 
              if path.suffix == ".json"]
  obj_paths = [path for path in paths if path.suffix == ".json"]
  for data, path in zip(obj_flat, obj_paths):
    path.write_text(json.dumps(data))

  # serialize arrays, avoid the zip(*[]) for clarity 
  arr_flat = [data for path, data in zip(paths, data_flat) if path.suffix == ""]
  arr_paths = [path for path in paths if path.suffix == ""]
  ts_specs = [get_tensorstore_spec(path) for path in arr_paths]
  
  async def _save_arrays():
    await asyncio.gather(*[serialization.async_serialize(arr, ts_spec) 
                        for (arr, ts_spec) in zip(arr_flat, ts_specs)])

  return await _save_arrays()

async def async_load(directory: str | PathLike, shardings: Sequence | None = None, 
                device: str | jax.Device | None = None) -> Any:
  root = epath.Path(directory)
  paths = [epath.Path(path) for path in root.glob("*") if 
           epath.Path(path).name != _PYTREEDEF_FILE]
  pytree = json.loads((root / _PYTREEDEF_FILE).read_text())
  values, pytreedef = tree.flatten(pytree)
  assert len(values) == len(paths), ("The number of read objects does not match"
                                     " the number of objects in the pytree.")
  missing_paths = [value for value in values 
                   if value not in set(path.stem for path in paths)]
  if len(missing_paths) > 0:
    raise ValueError(f"Values {missing_paths} are missing from the checkpoint"
                     " directory.")
  
  obj_paths = [path for path in paths if path.suffix == ".json"]
  objs = {path.stem: json.loads(path.read_text()) for path in obj_paths}
  
  arr_paths = [path for path in paths if path.suffix == ""]
  assert (shardings is None) or (device is None), ("Cannot specify shardings"
                                                   " and device at the same"
                                                   " time.")

  # missing sharding assumes we want to deserialize on host device
  if shardings is None:
    if device is None:
      device = jax.devices("cpu")[0]
    elif isinstance(device, str):
      device = jax.devices(device)[0]
    shardings = [SingleDeviceSharding(device) for _ in arr_paths]
  else:
    shardings = tree.flatten(shardings)[0]

  # deserialize 
  async def _deserialize_arrays():
    ts_specs = [get_tensorstore_spec(path) for path in arr_paths]
    arr_flat = await asyncio.gather(*[
      serialization.async_deserialize(sharding, ts_spec) 
      for (sharding, ts_spec) in zip(shardings, ts_specs)])
    arrs = {path.stem: arr for (path, arr) in zip(arr_paths, arr_flat)}

    arr_objs = objs | arrs
    filled_values = [arr_objs[value] for value in values]
    return tree.unflatten(pytreedef, filled_values)
  
  return await _deserialize_arrays()
  

def save(data: Any, directory: str | PathLike) -> None:
  def _async_save():
    asyncio.run(async_save(data=data, directory=directory))

  thread = threading.Thread(target=_async_save)
  thread.start()
  thread.join()

def load(directory: str | PathLike, shardings: Sequence | None = None, 
              device: str | jax.Device | None = None) -> Any:
  result = []
  def _async_load():
    result.append(asyncio.run(async_load(directory=directory, 
                                         shardings=shardings, device=device)))
  thread = threading.Thread(target=_async_load)
  thread.start()
  thread.join()
  return result[0]