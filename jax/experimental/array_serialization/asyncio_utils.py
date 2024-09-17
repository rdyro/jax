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

import asyncio
import functools
import logging
import threading


logger = logging.getLogger(__name__)

# Lifted from T5X.
class LimitInFlightBytes:
  """Limits in-flight bytes when reading/writing checkpoints per process."""

  def __init__(self, num_bytes):
    self._max_bytes = num_bytes
    self._available_bytes = num_bytes
    self._cv = asyncio.Condition(lock=asyncio.Lock())

  async def wait_for_bytes(self, requested_bytes):
    if requested_bytes > self._max_bytes:
      self._max_bytes = requested_bytes
      logger.warning("Requested more bytes than we reserved space for: %d > %d"
                     ". Increasing the limit to %d.", requested_bytes,
                     self._max_bytes, self._max_bytes)
    async with self._cv:
      await self._cv.wait_for(lambda: self._available_bytes > requested_bytes)
      self._available_bytes -= requested_bytes
      assert self._available_bytes >= 0

  async def release_bytes(self, requested_bytes):
    async with self._cv:
      self._available_bytes += requested_bytes
      assert self._available_bytes <= self._max_bytes
      self._cv.notify_all()


def _maybe_run_async_sync(name, async_fn):
  """Run async routine synchronously irrespective of the current environment."""

  def wrapped_fn(*args, **kw):
    retval, exception = [None], [None]

    def _run_in_thread():
      async def _async_thread_run():
        ret, exc = None, None
        try:
          ret = await async_fn(*args, **kw)
        except Exception as e:  # pylint: disable=broad-except
          exc = e
        return ret, exc

      retval[0], exception[0] = asyncio.run(_async_thread_run())

    t = threading.Thread(target=_run_in_thread)
    t.start()
    t.join()
    if exception[0] is not None:
      raise exception[0]  # pylint: disable=raising-bad-type
    return retval[0]

  functools.update_wrapper(wrapper=wrapped_fn, wrapped=async_fn)
  wrapped_fn.__name__ = name
  wrapped_fn.__qualname__ = name
  return wrapped_fn
