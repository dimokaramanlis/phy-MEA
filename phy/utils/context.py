# -*- coding: utf-8 -*-

"""Execution context that handles parallel processing and caching."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import OrderedDict
from functools import wraps
import inspect
import logging
import os
from pathlib import Path
from pickle import dump, load

import numpy as np

from phylib.utils._misc import save_json, load_json, load_pickle, save_pickle, _fullname
from .config import phy_config_dir, ensure_dir_exists

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Context
#------------------------------------------------------------------------------

def _nbytes(obj):
    """Rough estimate of the memory used by an object, used to bound the memcache."""
    if isinstance(obj, np.ndarray):
        return obj.nbytes + 128
    if isinstance(obj, dict):
        return 64 + sum(_nbytes(v) + 64 for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return 64 + sum(_nbytes(v) for v in obj)
    return 32


class _LRUMemcache(OrderedDict):
    """In-memory cache dictionary that evicts the least recently used items when the total size
    of the cached values exceeds `max_bytes`.

    NOTE: cluster ids are never reused, so every merge/split creates new cache entries while
    the entries of the deleted clusters are rarely used again. Without a bound, the memcache
    (which is also persisted and reloaded across sessions) grows indefinitely.

    """
    def __init__(self, max_bytes=None):
        super(_LRUMemcache, self).__init__()
        self.max_bytes = max_bytes
        self._sizes = {}
        self._total = 0

    def get(self, key, default=None):
        try:
            value = super(_LRUMemcache, self).__getitem__(key)
        except KeyError:
            return default
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self._total -= self._sizes.pop(key, 0)
        super(_LRUMemcache, self).__setitem__(key, value)
        size = _nbytes(value)
        self._sizes[key] = size
        self._total += size
        self._evict()

    def __delitem__(self, key):
        super(_LRUMemcache, self).__delitem__(key)
        self._total -= self._sizes.pop(key, 0)

    def _evict(self):
        if not self.max_bytes:
            return
        # Always keep the most recent item.
        while self._total > self.max_bytes and len(self) > 1:
            del self[next(iter(self))]

    def __reduce__(self):
        # NOTE: pickle as a plain dict, for compatibility with existing memcache files.
        return (dict, (dict(self),))


def _cache_methods(obj, memcached, cached):  # pragma: no cover
    for name in memcached:
        f = getattr(obj, name)
        setattr(obj, name, obj.context.memcache(f))

    for name in cached:
        f = getattr(obj, name)
        setattr(obj, name, obj.context.cache(f))


class Context(object):
    """Handle function disk and memory caching with joblib.

    Memcaching a function is used to save *in memory* the output of the function for all
    passed inputs. Input should be hashable. NumPy arrays are supported. The contents of the
    memcache in memory can be persisted to disk with `context.save_memcache()` and
    `context.load_memcache()`.

    Caching a function is used to save *on disk* the output of the function for all passed
    inputs. Input should be hashable. NumPy arrays are supported. This is to be preferred
    over memcache when the inputs or outputs are large, and when the computations are longer
    than loading the result from disk.

    Constructor
    -----------

    cache_dir : str
        The directory in which the cache will be created.
    verbose : int
        The verbosity level passed to joblib Memory.

    Examples
    --------

    ```python
    @context.memcache
    def my_function(x):
        return x * x

    @context.cache
    def my_function(x):
        return x * x
    ```

    """

    """Maximum cache size, in bytes."""
    cache_limit = 2 * 1024 ** 3  # 2 GB

    """Maximum size of the in-memory cache of every memcached function, in bytes."""
    memcache_limit = 128 * 1024 ** 2  # 128 MB

    def __init__(self, cache_dir, verbose=0):
        self.verbose = verbose
        # Make sure the cache directory exists.
        self.cache_dir = Path(cache_dir).expanduser()
        if not self.cache_dir.exists():
            logger.debug("Create cache directory `%s`.", self.cache_dir)
            os.makedirs(str(self.cache_dir))

        # Ensure the memcache directory exists.
        path = self.cache_dir / 'memcache'
        if not path.exists():
            path.mkdir()

        self._set_memory(self.cache_dir)
        self._memcache = {}

    def _set_memory(self, cache_dir):
        """Create the joblib Memory instance."""

        # Try importing joblib.
        try:
            from joblib import Memory
            self._memory = Memory(
                location=self.cache_dir, mmap_mode=None, verbose=self.verbose,
                bytes_limit=self.cache_limit)
            logger.debug("Initialize joblib cache dir at `%s`.", self.cache_dir)
            logger.debug("Reducing the size of the cache if needed.")
            self._memory.reduce_size()
        except ImportError:  # pragma: no cover
            logger.warning(
                "Joblib is not installed. Install it with `conda install joblib`.")
            self._memory = None

    def cache(self, f):
        """Cache a function using the context's cache directory."""
        if self._memory is None:  # pragma: no cover
            logger.debug("Joblib is not installed: skipping caching.")
            return f
        assert f
        # NOTE: discard self in instance methods.
        if 'self' in inspect.getfullargspec(f).args:
            ignore = ['self']
        else:
            ignore = None
        disk_cached = self._memory.cache(f, ignore=ignore)
        return disk_cached

    def load_memcache(self, name):
        """Load the memcache from disk (pickle file), if it exists."""
        path = self.cache_dir / 'memcache' / (name + '.pkl')
        cache = _LRUMemcache(max_bytes=self.memcache_limit)
        if path.exists():
            logger.debug("Load memcache for `%s`.", name)
            try:
                with open(str(path), 'rb') as fd:
                    # NOTE: the oldest items are evicted if the saved memcache is too large.
                    for key, value in load(fd).items():
                        cache[key] = value
            except Exception as e:  # pragma: no cover
                logger.debug("Could not load memcache for `%s`: %s.", name, e)
        self._memcache[name] = cache
        return cache

    def save_memcache(self):
        """Save the memcache to disk using pickle."""
        for name, cache in self._memcache.items():
            path = self.cache_dir / 'memcache' / (name + '.pkl')
            logger.debug("Save memcache for `%s`.", name)
            with open(str(path), 'wb') as fd:
                dump(cache, fd)

    def memcache(self, f):
        """Cache a function in memory using an internal dictionary."""
        name = _fullname(f)
        cache = self.load_memcache(name)

        @wraps(f)
        def memcached(*args, **kwargs):
            """Cache the function in memory."""
            # The arguments need to be hashable. Much faster than using hash().
            h = args
            out = cache.get(h, None)
            if out is None:
                out = f(*args, **kwargs)
                cache[h] = out
            return out
        return memcached

    def _get_path(self, name, location, file_ext='.json'):
        """Get the path to the cache file."""
        if location == 'local':
            return self.cache_dir / (name + file_ext)
        elif location == 'global':
            return phy_config_dir() / (name + file_ext)

    def save(self, name, data, location='local', kind='json'):
        """Save a dictionary in a JSON/pickle file within the cache directory.

        Parameters
        ----------

        name : str
            The name of the object to save to disk.
        data : dict
            Any serializable dictionary that will be persisted to disk.
        location : str
            Can be `local` or `global`.
        kind : str
            Can be `json` or `pickle`.

        """
        file_ext = '.json' if kind == 'json' else '.pkl'
        path = self._get_path(name, location, file_ext=file_ext)
        ensure_dir_exists(path.parent)
        logger.debug("Save data to `%s`.", path)
        if kind == 'json':
            save_json(path, data)
        else:
            save_pickle(path, data)

    def load(self, name, location='local'):
        """Load a dictionary saved in the cache directory.

        Parameters
        ----------

        name : str
            The name of the object to save to disk.
        location : str
            Can be `local` or `global`.

        """
        path = self._get_path(name, location, file_ext='.json')
        if path.exists():
            return load_json(path)
        path = self._get_path(name, location, file_ext='.pkl')
        if path.exists():
            return load_pickle(path)
        logger.debug("The file `%s` doesn't exist.", path)
        return {}

    def __getstate__(self):
        """Make sure that this class is picklable."""
        state = self.__dict__.copy()
        state['_memory'] = None
        return state

    def __setstate__(self, state):
        """Make sure that this class is picklable."""
        self.__dict__ = state
        # Recreate the joblib Memory instance.
        self._set_memory(state['cache_dir'])
