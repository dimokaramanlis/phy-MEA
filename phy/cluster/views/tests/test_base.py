# -*- coding: utf-8 -*-

"""Test scatter view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.utils import emit
from phy.utils.color import selected_cluster_color, colormaps
from ..base import BaseColorView, ManualClusteringView
from . import _stop_and_close


#------------------------------------------------------------------------------
# Test manual clustering view
#------------------------------------------------------------------------------

class MyView(BaseColorView, ManualClusteringView):
    def plot(self, **kwargs):
        for i in range(len(self.cluster_ids)):
            self.canvas.scatter(pos=.25 * np.random.randn(100, 2), color=selected_cluster_color(i))

    @property
    def status(self):
        return 'hello'


def test_manual_clustering_view_1(qtbot, tempdir):
    v = MyView()
    v.canvas.show()
    # qtbot.addWidget(v.canvas)
    v.on_select(cluster_ids=[0, 1])

    v.set_state({'auto_update': False})
    assert v.auto_update is False

    qtbot.wait(10)

    path = v.screenshot(dir=tempdir)
    qtbot.wait(10)

    assert str(path).startswith(str(tempdir))
    assert path.exists()

    _stop_and_close(qtbot, v)


def test_manual_clustering_view_2(qtbot, gui):
    v = MyView()
    v.canvas.show()
    v.add_color_scheme(
        lambda cid: cid, name='myscheme', colormap=colormaps.rainbow, cluster_ids=[0, 1])
    v.attach(gui)

    class Supervisor(object):
        pass

    emit('select', Supervisor(), cluster_ids=[0, 1])

    v.actions.get('Change color scheme to myscheme').trigger()
    v.next_color_scheme()
    v.previous_color_scheme()
    assert v.get_cluster_colors([0, 1]).shape == (2, 4)

    qtbot.wait(200)
    # qtbot.stop()
    v.canvas.close()
    v.actions.close()
    qtbot.wait(100)


def test_unconnect_objects():
    import gc
    import weakref
    from functools import partial
    from phylib.utils import connect, emit
    from phylib.utils.event import _EVENT
    from ..base import _unconnect_objects

    class Obj(object):
        def on_select(self, sender, *args):
            pass

    view, canvas, other = Obj(), Obj(), Obj()
    n = len(_EVENT._callbacks)
    calls = []

    connect(view.on_select, event='select')
    connect(partial(view.on_select, 1), event='select')
    connect(lambda sender: calls.append(view), event='cluster', sender=other)
    connect(lambda sender: None, event='zoom', sender=canvas)
    # Unrelated callback: kept.
    connect(lambda sender: calls.append('kept'), event='cluster', sender=other)
    assert len(_EVENT._callbacks) == n + 5

    ref = weakref.ref(view)
    _unconnect_objects(view, canvas)
    assert len(_EVENT._callbacks) == n + 1
    emit('cluster', other)
    assert calls == ['kept']

    del view
    gc.collect()
    assert ref() is None
    _unconnect_objects(other)
    assert len(_EVENT._callbacks) == n
