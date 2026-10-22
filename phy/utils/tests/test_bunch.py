# -*- coding: utf-8 -*-

"""Test the cycle-free Bunch."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import gc
import pickle
import weakref

import numpy as np
from pytest import raises

import phy  # noqa: patches Bunch
from phylib.utils import Bunch
from phylib.utils._types import Bunch as _Bunch


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_bunch_patched():
    assert Bunch is _Bunch
    assert Bunch._phy_cycle_free


def test_bunch_attributes():
    b = Bunch(a=1)
    b.b = 2
    assert b == {'a': 1, 'b': 2}
    assert b.a == 1
    assert b['b'] == 2
    del b.b
    assert 'b' not in b
    with raises(AttributeError):
        b.b
    with raises(AttributeError):
        del b.b
    assert getattr(b, 'c', 3) == 3

    c = b.copy()
    assert isinstance(c, Bunch)
    assert c == b


def test_bunch_property():
    class MyBunch(Bunch):
        @property
        def double(self):
            return 2 * self.x

    b = MyBunch(x=1)
    assert b.double == 2
    with raises(AttributeError):
        b.double = 3
    assert 'double' not in b


def test_bunch_pickle():
    b = Bunch(a=1, arr=np.arange(3))
    c = pickle.loads(pickle.dumps(b))
    assert isinstance(c, Bunch)
    assert c.a == 1
    assert list(c.arr) == [0, 1, 2]


def test_bunch_no_cycle():
    class Arr(np.ndarray):
        pass

    arr = np.zeros(10).view(Arr)
    ref = weakref.ref(arr)
    b = Bunch(arr=arr)
    del arr
    gc.disable()
    try:
        # The Bunch and its array are freed without the garbage collector.
        del b
        assert ref() is None
    finally:
        gc.enable()
