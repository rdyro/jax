# Copyright 2023 The JAX Authors.
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

"""Extensions to automatically generated TPU dialect bindings."""


class TraceOp:
  """An extension to the automatically generated TraceOp bindings."""

  def __init__(self, results, message, level, *, loc=None, ip=None):
    super().__init__(results, message, level, loc=loc, ip=ip)
    self.regions[0].blocks.append(*[])  # Append the block.

  @property
  def body(self):
    return self.regions[0].blocks[0]


class RegionOp:
  """An extension to the automatically generated RegionOp bindings."""

  def __init__(self, *, loc=None, ip=None):
    super().__init__([], loc=loc, ip=ip)
    self.regions[0].blocks.append()  # Append the block.

  @property
  def body(self):
    return self.regions[0].blocks[0]
