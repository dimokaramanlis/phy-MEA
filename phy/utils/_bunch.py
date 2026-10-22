# -*- coding: utf-8 -*-

"""Make phylib's Bunch free of reference cycles."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from phylib.utils._types import Bunch


#------------------------------------------------------------------------------
# Cycle-free Bunch
#------------------------------------------------------------------------------

def _init(self, *args, **kwargs):
    dict.__init__(self, *args, **kwargs)


def _getattr(self, name):
    # NOTE: only called when the normal attribute lookup fails (class attributes, methods and
    # properties take precedence, as with the original implementation).
    try:
        return self[name]
    except KeyError:
        raise AttributeError(name)


def _setattr(self, name, value):
    # Respect data descriptors (properties) defined in subclasses.
    if hasattr(getattr(type(self), name, None), '__set__'):
        object.__setattr__(self, name, value)
    else:
        self[name] = value


def _delattr(self, name):
    try:
        del self[name]
    except KeyError:
        raise AttributeError(name)


def _setstate(self, state):
    # NOTE: Bunch instances pickled with the original implementation have their items as state.
    if state:
        self.update(state)


def patch_bunch():
    """Make phylib's `Bunch` free of reference cycles.

    The original `Bunch.__init__()` does `self.__dict__ = self`: every Bunch references itself,
    so it is never freed by reference counting, only by the (full) cyclic garbage collection.
    phy wraps all data computed for the views (waveforms, features, amplitudes, visual data...)
    in Bunch instances: their arrays stay in memory until a full collection, which becomes rarer
    and slower as the session goes on. Memory usage grows by up to several GB with large
    datasets, and the GUI slows down.

    The class is patched in place, so that all modules that already imported it are affected.

    """
    if getattr(Bunch, '_phy_cycle_free', False):
        return
    Bunch.__init__ = _init
    Bunch.__getattr__ = _getattr
    Bunch.__setattr__ = _setattr
    Bunch.__delattr__ = _delattr
    Bunch.__setstate__ = _setstate
    Bunch._phy_cycle_free = True
