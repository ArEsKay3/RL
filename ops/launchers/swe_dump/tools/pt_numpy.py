"""Load a torch.save() zip archive into numpy arrays without importing torch.

Only the subset of the pickle protocol that torch.save emits for plain tensors
and Python containers is supported (storage persistent ids, _rebuild_tensor_v2,
OrderedDict, torch.Size). Tensors are returned as numpy arrays; everything else
as the corresponding Python object.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from typing import Any

import numpy as np

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "BFloat16Storage": np.uint16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _Storage:
    def __init__(self, dtype: np.dtype, raw: bytes):
        self.array = np.frombuffer(raw, dtype=dtype)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    size = tuple(int(s) for s in size)
    stride = tuple(int(s) for s in stride)
    base = storage.array
    if not size:
        return base[storage_offset : storage_offset + 1].reshape(())
    itemsize = base.dtype.itemsize
    return np.lib.stride_tricks.as_strided(
        base[storage_offset:],
        shape=size,
        strides=tuple(s * itemsize for s in stride),
        writeable=False,
    )


def _rebuild_parameter(data, requires_grad, backward_hooks):
    return data


class _Size(tuple):
    pass


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh, archive: zipfile.ZipFile, prefix: str):
        super().__init__(fh)
        self._archive = archive
        self._prefix = prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _STORAGE_DTYPES:
            return name
        if module == "torch" and name == "Size":
            return _Size
        if module == "collections" and name == "OrderedDict":
            import collections

            return collections.OrderedDict
        if module == "torch.storage" and name == "_load_from_bytes":
            raise pickle.UnpicklingError("legacy storage format not supported")
        return super().find_class(module, name)

    def persistent_load(self, pid):
        kind, storage_type, key, location, numel = pid[:5]
        assert kind == "storage", kind
        dtype = np.dtype(_STORAGE_DTYPES[storage_type if isinstance(storage_type, str) else storage_type.__name__])
        raw = self._archive.read(f"{self._prefix}data/{key}")
        return _Storage(dtype, raw)


def load(path: str) -> Any:
    archive = zipfile.ZipFile(path)
    names = archive.namelist()
    pkl = next(n for n in names if n.endswith("data.pkl"))
    prefix = pkl[: -len("data.pkl")]
    with archive.open(pkl) as fh:
        return _Unpickler(io.BytesIO(fh.read()), archive, prefix).load()


if __name__ == "__main__":
    import sys

    obj = load(sys.argv[1])
    for k, v in obj.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: ndarray {v.dtype} {v.shape}")
        elif isinstance(v, list):
            print(f"{k}: list[{len(v)}] {str(v[:2])[:80]}")
        else:
            print(f"{k}: {type(v).__name__} {str(v)[:80]}")
