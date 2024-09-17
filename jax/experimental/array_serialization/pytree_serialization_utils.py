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

import os
import collections
import importlib
import logging
from uuid import uuid4, UUID
from types import ModuleType
from typing import Any
import pickle
import io
import zipfile
import threading
import asyncio

import jax
from jax.tree_util import PyTreeDef, default_registry, treedef_is_leaf
from jax.experimental import multihost_utils
from jax._src.util import safe_zip
import numpy as np

PickleModule = ModuleType

logger = logging.getLogger(__name__)

_NOT_SERIALIZABLE_WITHOUT_PICKLE = (
  "Node data `{}` is not serializable without a pickle_module.")
_TREE_REPR_KEY = "__jax_tree_repr"
_LEAF_IDS_KEY = "__jax_leaf_ids"


def _broadcast_uuids(uuids: list[UUID]) -> list[UUID]:
  """Broadcast a list of UUIDs to all hosts to ensure agreement."""
  encoded = np.stack([np.frombuffer(x.bytes, dtype=np.int32) for x in uuids])
  encoded = multihost_utils.broadcast_one_to_all(encoded)
  return [UUID(bytes=x.tobytes()) for x in np.array(encoded)]


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

def _count_custom_data(node: PyTreeDef,
                       pickle_module: PickleModule | None = None) -> int:
  if (node.node_data() is None
      or _cls2typerepr(node.node_data()[0]) in _EXTENDED_NODE_TYPES_MAP):
    own_count = 0
  else:
    if pickle_module is None:
      raise ValueError(
        _NOT_SERIALIZABLE_WITHOUT_PICKLE.format(node.node_data()))
    own_count = 1
  return own_count + sum(_count_custom_data(child, pickle_module)
                         for child in node.children())

def _encode_node_data(node_data, node_data_store: dict[str, Any],
                      pickle_module: PickleModule | None,
                      node_uuids: list[UUID]):
  if node_data is None:
    return "leaf", None
  elif _cls2typerepr(node_data[0]) in _EXTENDED_NODE_TYPES_MAP:
    return (_cls2typerepr(node_data[0]), node_data[1])
  else:
    # the fallback case
    node_data_key = str(node_uuids.pop())
    if pickle_module is None:
      raise ValueError(_NOT_SERIALIZABLE_WITHOUT_PICKLE.format(node_data))
    # we pickle node_data which includes **both** the class and data
    node_data_store[node_data_key] = pickle_module.dumps(node_data)
    return (_cls2typerepr(node_data[0]), node_data_key)

def _decode_extended_type(typerepr: str, node_data: Any, best_effort: bool):
  """Given a type name like `flax.core.frozen_dict` try to return the class."""
  module_name, cls_name = typerepr.rsplit(".", 1)
  missing_attribute = False
  try:
    module = importlib.import_module(module_name)
    if not hasattr(module, cls_name):
      missing_attribute = True
      raise ImportError
    node_cls = getattr(module, cls_name)
  except ImportError as exc:
    missing_mod_msg = f"Could not find module `{module_name}`."
    missing_cls_msg = (f"Coud not find class `{cls_name}` in module"
                        f" `{module_name}`.")
    if best_effort:
      del exc  # explicitly ignore the exception
      msg = f" Falling back to a list of children since {best_effort=}."
      logger.warning((missing_mod_msg if missing_attribute
                      else missing_cls_msg) + msg)
      node_cls, node_data = (list, None)
    else:
      raise ImportError(missing_mod_msg if missing_attribute
                        else missing_cls_msg) from exc
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
      try:
        assert pickle_module is not None
        return pickle_module.loads(node_data_store[node_data_ref])
      except (pickle.UnpicklingError, AttributeError, AssertionError
              ) as pickle_error:
        # fallback
        if best_effort:
          logging.warning("Unrecognized data type `%s` we'll do our best and"
                          " just return a list of children", node_type)
          return (list, None)
        else:
          raise pickle_error
    else:
      if best_effort:
        logging.warning("Unrecognized data type `%s` we'll do our best and just"
                        " return a list of children", node_type)
        return (list, None)
      else:
        raise KeyError(f"Node reference `{node_data_ref}` missing from node"
                       " data store.")


def _serialize_pytree_helper(node, node_data_store: dict[str, Any],
                             pickle_module: PickleModule | None,
                             leaf_uuids: list[UUID], node_uuids: list[UUID]):
  node_repr = dict()
  node_repr["node_type"], node_repr["node_data_ref"] = _encode_node_data(
    node.node_data(), node_data_store, pickle_module, node_uuids)
  # this encodes if-and-only-if using not and xor
  assert not ((len(node.children()) == 0) ^ treedef_is_leaf(node))

  if not treedef_is_leaf(node):
    children, leaf_ids = safe_zip(*[
      _serialize_pytree_helper(child, node_data_store=node_data_store,
                               pickle_module=pickle_module,
                               leaf_uuids=leaf_uuids, node_uuids=node_uuids)
      for child in node.children()])
    node_repr["children"] = children
  else:
    node_repr["children"] = []
    assert node.num_leaves in (0, 1), (
      "We only support 1 or 0 leaves (?) is this a leaf")
    # leaf_id = str(uuid4()) if node.num_leaves == 1 else None
    leaf_id = str(leaf_uuids.pop()) if node.num_leaves == 1 else None
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
  def deserialize_pytree(rawtree: dict[str, Any],
                         node_data_store: dict[str, Any] | None = None):
    raise NotImplementedError

class PermissivePyTreeSerialization(AbstractPyTreeSerialization):
  @staticmethod
  def serialize_pytree(node, pickle_module: PickleModule | None = None):
    node_data_store = dict()
    # preallocate uuids for all leaves
    leaf_uuids = [uuid4() for _ in range(node.num_leaves)]
    node_uuids_len = _count_custom_data(node, pickle_module)
    node_uuids = [uuid4() for _ in range(node_uuids_len)]
    if jax.process_count() > 1:
      all_uuids = _broadcast_uuids(leaf_uuids + node_uuids)
      leaf_uuids = all_uuids[:len(leaf_uuids)]
      node_uuids = all_uuids[len(leaf_uuids):]

    tree_repr, leaf_ids = _serialize_pytree_helper(node, node_data_store,
                                                   pickle_module, leaf_uuids,
                                                   node_uuids)
    node_repr = dict()
    node_repr[_TREE_REPR_KEY] = tree_repr
    node_repr[_LEAF_IDS_KEY] = jax.tree.flatten(leaf_ids)[0]
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
    return jax.tree.unflatten(pt, leaf_ids)

class StrictPyTreeSerialization(AbstractPyTreeSerialization):
  @staticmethod
  def serialize_pytree(node, pickle_module: PickleModule | None = None):
    del pickle_module
    leaf_ids = [str(uuid4()) for _ in range(node.num_leaves)]
    _ = PyTreeDef.serialize_using_proto(node)  # this raises a useful error
    # alternatively: assert _is_pytree_strict_serializable(node)
    node_repr = jax.tree.unflatten(node, leaf_ids)
    return node_repr, leaf_ids, dict()

  @staticmethod
  def deserialize_pytree(rawtree: PyTreeDef,
                         node_data_store: dict[str, Any] | None = None,
                         pickle_module: PickleModule | None = None,
                         best_effort: bool = False) -> PyTreeDef:
    del node_data_store
    del pickle_module
    del best_effort
    return rawtree


class InMemoryZip:
  def __init__(self, read_mode: bool | None = None,
               data: bytes | None = None, **kw):
    if data is not None:
      self.read_mode = read_mode if read_mode is not None else True
      self.buffer = io.BytesIO(data)
      assert self.read_mode
    else:
      self.read_mode = read_mode if read_mode is not None else False
      self.buffer = io.BytesIO()
    self.opts = dict({"mode": "r" if self.read_mode else "w",
                      "compression": zipfile.ZIP_DEFLATED, "allowZip64": True},
                     **kw)
    self.zipfile = zipfile.ZipFile(self.buffer, "w"
                                   if not self.read_mode else "r")
    self.thread_lock = threading.Lock()
    self.async_lock = asyncio.Lock()
    self._closed = False

  def tobytes(self) -> bytes:
    assert not self._closed
    with self.thread_lock:
      self._closed = True
      self.zipfile.close()
      return self.buffer.getvalue()

  def keys(self) -> list[str]:
    assert not self._closed and self.read_mode
    return self.zipfile.namelist()

  async def async_tobytes(self) -> bytes:
    assert not self._closed
    async with self.async_lock:
      self.zipfile.close()
      self._closed = True
      return self.buffer.getvalue()

  def write(self, filename: str | os.PathLike[str], data: bytes | str) -> None:
    assert not self._closed
    with self.thread_lock:
      self.zipfile.writestr(str(filename), data)

  def read(self, filename: str | os.PathLike[str]) -> bytes:
    assert not self._closed and self.read_mode
    return self.zipfile.read(str(filename))

  async def async_write(self, filename: str | os.PathLike[str],
                        data: bytes | str) -> None:
    assert not self._closed
    async with self.async_lock:
      self.zipfile.writestr(str(filename), data)

  async def async_read(self, filename: str | os.PathLike[str]) -> bytes:
    assert not self._closed and self.read_mode
    return self.zipfile.read(str(filename))
