# -*- coding: utf-8 -*-

"""Clustering structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import numpy as np

from phylib.utils._types import _as_array, _is_array_like
from phylib.io.array import _unique, _spikes_in_clusters, _spikes_per_cluster
from ._utils import UpdateInfo
from ._history import History
from phylib.utils.event import emit

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Clustering class
#------------------------------------------------------------------------------

class _CompactIds(object):
    """Compact storage, in the undo stack, of per-spike cluster ids: the unique cluster ids,
    and the index of every spike's cluster in the smallest possible integer type (few
    clusters are involved in an action, so this is usually 1 byte per spike instead of 8)."""
    __slots__ = ('unique', 'index')

    def __init__(self, unique, index):
        self.unique = unique
        self.index = index


def _compact_ids(spike_clusters):
    """Compact per-spike cluster ids for storage in the undo stack."""
    spike_clusters = np.asarray(spike_clusters)
    if spike_clusters.size <= 1:
        return spike_clusters
    unique, index = np.unique(spike_clusters, return_inverse=True)
    dtype = np.uint8 if len(unique) <= 256 else np.uint16 if len(unique) <= 65536 else np.int64
    return _CompactIds(unique, index.astype(dtype))


def _compact_spike_ids(spike_ids, n_spikes):
    """Store spike ids with 4 bytes instead of 8 when possible."""
    if n_spikes < 2 ** 31:
        return np.asarray(spike_ids).astype(np.int32)
    return spike_ids


def _expand_ids(obj):
    """Inverse of `_compact_ids()` and `_compact_spike_ids()`."""
    if isinstance(obj, _CompactIds):
        return obj.unique[obj.index]
    if isinstance(obj, np.ndarray) and obj.dtype == np.int32:
        return obj.astype(np.int64)
    return obj


def _extend_spikes(spike_ids, spike_clusters, spikes_per_cluster=None):
    """Return all spikes belonging to the clusters containing the specified
    spikes."""
    # We find the spikes belonging to modified clusters.
    # What are the old clusters that are modified by the assignment?
    old_spike_clusters = spike_clusters[spike_ids]
    unique_clusters = _unique(old_spike_clusters)
    # Now we take all spikes from these clusters.
    if spikes_per_cluster is not None:
        # OPTIM: avoid going through all spikes.
        changed_spike_ids = np.sort(np.concatenate(
            [_as_array(spikes_per_cluster[c]) for c in unique_clusters])).astype(np.int64)
    else:
        changed_spike_ids = _spikes_in_clusters(spike_clusters, unique_clusters)
    # These are the new spikes that need to be reassigned.
    extended_spike_ids = np.setdiff1d(changed_spike_ids, spike_ids, assume_unique=True)
    return extended_spike_ids


def _concatenate_spike_clusters(*pairs):
    """Concatenate a list of pairs (spike_ids, spike_clusters)."""
    pairs = [(_as_array(x), _as_array(y)) for (x, y) in pairs]
    concat = np.vstack([np.hstack((x[:, None], y[:, None])) for x, y in pairs])
    reorder = np.argsort(concat[:, 0])
    concat = concat[reorder, :]
    return concat[:, 0].astype(np.int64), concat[:, 1].astype(np.int64)


def _extend_assignment(
        spike_ids, old_spike_clusters, spike_clusters_rel, new_cluster_id,
        spikes_per_cluster=None):
    # 1. Add spikes that belong to modified clusters.
    # 2. Find new cluster ids for all changed clusters.

    old_spike_clusters = _as_array(old_spike_clusters)
    spike_ids = _as_array(spike_ids)

    assert isinstance(spike_clusters_rel, (list, np.ndarray))
    spike_clusters_rel = _as_array(spike_clusters_rel)
    assert spike_clusters_rel.min() >= 0

    # We renumber the new cluster indices.
    new_spike_clusters = (spike_clusters_rel + (new_cluster_id - spike_clusters_rel.min()))

    # We find the spikes belonging to modified clusters.
    extended_spike_ids = _extend_spikes(
        spike_ids, old_spike_clusters, spikes_per_cluster=spikes_per_cluster)
    if len(extended_spike_ids) == 0:
        return spike_ids, new_spike_clusters

    # We take their clusters.
    extended_spike_clusters = old_spike_clusters[extended_spike_ids]
    # Use relative numbers in extended_spike_clusters.
    _, extended_spike_clusters = np.unique(extended_spike_clusters, return_inverse=True)
    # Generate new cluster numbers.
    k = new_spike_clusters.max() + 1
    extended_spike_clusters += (k - extended_spike_clusters.min())

    # Finally, we concatenate spike_ids and extended_spike_ids.
    return _concatenate_spike_clusters(
        (spike_ids, new_spike_clusters), (extended_spike_ids, extended_spike_clusters))


def _assign_update_info(spike_ids, old_spike_clusters, new_spike_clusters):
    old_clusters = _unique(old_spike_clusters)
    new_clusters = _unique(new_spike_clusters)
    largest_old_cluster = np.bincount(old_spike_clusters).argmax()
    # Unique (old, new) pairs, vectorized (a Python loop over all spikes is slow).
    old_spike_clusters = np.asarray(old_spike_clusters, dtype=np.int64)
    new_spike_clusters = np.asarray(new_spike_clusters, dtype=np.int64)
    k = int(new_spike_clusters.max()) + 1
    pairs = np.unique(old_spike_clusters * k + new_spike_clusters)
    descendants = [(int(p // k), int(p % k)) for p in pairs]
    update_info = UpdateInfo(
        description='assign',
        spike_ids=np.asarray(spike_ids),
        spike_clusters=np.asarray(new_spike_clusters),
        added=list(new_clusters),
        deleted=list(old_clusters),
        descendants=descendants,
        largest_old_cluster=int(largest_old_cluster),
    )
    return update_info


class Clustering(object):
    """Handle cluster changes in a set of spikes.

    Constructor
    -----------

    spike_clusters : array-like
        Spike-cluster assignments, giving the cluster id of every spike.
    new_cluster_id : int
        Cluster id that is not used yet (and not used in the cache if there is one). We need to
        ensure that cluster ids are unique and not reused in a given session.
    spikes_per_cluster : dict
        Dictionary mapping each cluster id to the spike ids belonging to it. This is recomputed
        if not given. This object may take a while to compute, so it may be cached and passed
        to the constructor.

    Features
    --------

    * List of clusters appearing in a `spike_clusters` array
    * Dictionary of spikes per cluster
    * Merge
    * Split and assign
    * Undo/redo stack

    Notes
    -----

    The undo stack works by keeping the list of all spike cluster changes
    made successively, together with the previous assignment of the affected
    spikes. Undoing consists of restoring that previous assignment.

    UpdateInfo
    ----------

    Most methods of this class return an `UpdateInfo` instance. This object
    contains information about the clustering changes done by the operation.
    This object is used throughout the `phy.cluster.manual` package to let
    different classes know about clustering changes.

    `UpdateInfo` is a dictionary that also supports dot access (`Bunch` class).

    """

    def __init__(self, spike_clusters, new_cluster_id=None,
                 spikes_per_cluster=None):
        super(Clustering, self).__init__()
        # The stack contains (spike_ids, new_spike_clusters, old_spike_clusters, undo_state)
        # tuples. Undoing an action simply restores the old assignment of the affected spikes.
        self._undo_stack = History(base_item=(None, None, None, None))
        # Spike -> cluster mapping.
        self._spike_clusters = _as_array(spike_clusters)
        self._spikes_per_cluster = {}
        self._n_spikes = len(self._spike_clusters)
        self._spike_ids = np.arange(self._n_spikes).astype(np.int64)
        # We can pass the precomputed spikes_per_cluster dictionary for
        # performance reasons.
        self._update_cluster_ids(to_add=spikes_per_cluster)
        self._new_cluster_id_0 = int(new_cluster_id or self._spike_clusters.max() + 1)
        self._new_cluster_id = self._new_cluster_id_0
        assert self._new_cluster_id >= 0
        assert np.all(self._spike_clusters < self._new_cluster_id)

    def reset(self):
        """Reset the clustering to the original clustering.

        All changes are lost.

        """
        # NOTE: we undo all actions instead of keeping a full copy of the original
        # spike_clusters array in memory.
        while self._undo_stack.current_position > 0:
            spike_ids, _, old_spike_clusters, _ = self._undo_stack.back()
            self._do_assign(_expand_ids(spike_ids), _expand_ids(old_spike_clusters))
        self._undo_stack.clear(base_item=(None, None, None, None))
        self._new_cluster_id = self._new_cluster_id_0

    @property
    def spike_clusters(self):
        """A n_spikes-long vector containing the cluster ids of all spikes."""
        return self._spike_clusters

    @property
    def spikes_per_cluster(self):
        """A dictionary {cluster_id: spike_ids}."""
        return self._spikes_per_cluster

    @property
    def cluster_ids(self):
        """Ordered list of ids of all non-empty clusters."""
        return self._cluster_ids

    def new_cluster_id(self):
        """Generate a brand new cluster id.

        Note
        ----

        This new id strictly increases after an undo + new action,
        meaning that old cluster ids are *not* reused. This ensures that
        any cluster_id-based cache will always be valid even after undo
        operations (i.e. no need for explicit cache invalidation in this case).

        """
        return self._new_cluster_id

    @property
    def n_clusters(self):
        """Total number of clusters."""
        return len(self.cluster_ids)

    @property
    def n_spikes(self):
        """Number of spikes."""
        return self._n_spikes

    @property
    def spike_ids(self):
        """Array of all spike ids."""
        return self._spike_ids

    def spikes_in_clusters(self, clusters):
        """Return the array of spike ids belonging to a list of clusters."""
        return _spikes_in_clusters(self.spike_clusters, clusters)

    # Actions
    #--------------------------------------------------------------------------

    def _spikes_per_cluster_coherent(self):
        """Check that spikes_per_cluster (which may come from the cache) matches spike_clusters.
        This is important because merges rely on spikes_per_cluster."""
        spc = self._spikes_per_cluster
        if not np.all(np.in1d(self._cluster_ids, sorted(spc))):
            return False
        sc = self._spike_clusters
        n = 0
        for clu in self._cluster_ids:
            spk = _as_array(spc[clu])
            n += len(spk)
            if len(spk) and (spk.max() >= len(sc) or np.any(sc[spk] != clu)):
                return False
        return n == len(sc)

    def _update_cluster_ids(self, to_remove=None, to_add=None, incremental=False):
        # Update the list of non-empty cluster ids.
        if incremental:
            # OPTIM: after an action, the removed clusters are empty and the added clusters
            # are new, so there is no need to go through the whole spike_clusters array.
            cluster_ids = np.setdiff1d(self._cluster_ids, np.asarray(list(to_remove)))
            self._cluster_ids = np.union1d(
                cluster_ids, np.asarray(list(to_add), dtype=cluster_ids.dtype))
        else:
            self._cluster_ids = _unique(self._spike_clusters)
        # Clusters to remove.
        if to_remove is not None:
            for clu in to_remove:
                self._spikes_per_cluster.pop(clu, None)
        # Clusters to add.
        if to_add:
            for clu, spk in to_add.items():
                self._spikes_per_cluster[clu] = spk
        if incremental:
            return
        # If spikes_per_cluster is invalid, recompute the entire
        # spikes_per_cluster array.
        coherent = self._spikes_per_cluster_coherent()
        if not coherent:
            logger.debug("Recompute spikes_per_cluster manually: this might take a while.")
            sc = self._spike_clusters
            self._spikes_per_cluster = _spikes_per_cluster(sc)

    def _do_assign(self, spike_ids, new_spike_clusters):
        """Make spike-cluster assignments after the spike selection has
        been extended to full clusters."""

        # Ensure spike_clusters has the right shape.
        spike_ids = _as_array(spike_ids)
        if len(new_spike_clusters) == 1 and len(spike_ids) > 1:
            new_spike_clusters = np.ones(len(spike_ids), dtype=np.int64) * new_spike_clusters[0]
        old_spike_clusters = self._spike_clusters[spike_ids]

        assert len(spike_ids) == len(old_spike_clusters)
        assert len(new_spike_clusters) == len(spike_ids)

        # Update the spikes per cluster structure.
        old_clusters = _unique(old_spike_clusters)

        # NOTE: shortcut to a merge if this assignment is effectively a merge
        # i.e. if all spikes are assigned to a single cluster.
        # The fact that spike selection has been previously extended to
        # whole clusters is critical here.
        new_clusters = _unique(new_spike_clusters)
        if len(new_clusters) == 1:
            return self._do_merge(spike_ids, old_clusters, new_clusters[0])

        # We return the UpdateInfo structure.
        up = _assign_update_info(spike_ids, old_spike_clusters, new_spike_clusters)

        # We update the new cluster id (strictly increasing during a session).
        self._new_cluster_id = max(self._new_cluster_id, max(up.added) + 1)

        # We make the assignments.
        self._spike_clusters[spike_ids] = new_spike_clusters
        # OPTIM: we update spikes_per_cluster manually.
        new_spc = _spikes_per_cluster(new_spike_clusters, spike_ids)
        self._update_cluster_ids(to_remove=old_clusters, to_add=new_spc, incremental=True)
        up.all_cluster_ids = list(self.cluster_ids)
        return up

    def _do_merge(self, spike_ids, cluster_ids, to):

        # Create the UpdateInfo instance here.
        descendants = [(cluster, to) for cluster in cluster_ids]
        largest_old_cluster = np.bincount(self.spike_clusters[spike_ids]).argmax()
        up = UpdateInfo(
            description='merge',
            # NOTE: keep the array, a Python list takes several times more memory.
            spike_ids=spike_ids,
            added=[to],
            deleted=list(cluster_ids),
            descendants=descendants,
            largest_old_cluster=largest_old_cluster,
        )

        # We update the new cluster id (strictly increasing during a session).
        self._new_cluster_id = max(max(up.added) + 1, self._new_cluster_id)

        # Assign the clusters.
        self.spike_clusters[spike_ids] = to
        # Update the list of non-empty cluster ids.
        # OPTIM: we update spikes_per_cluster manually.
        self._update_cluster_ids(
            to_remove=cluster_ids, to_add={to: spike_ids}, incremental=True)
        up.all_cluster_ids = list(self.cluster_ids)
        return up

    def merge(self, cluster_ids, to=None):
        """Merge several clusters to a new cluster.

        Parameters
        ----------

        cluster_ids : array-like
            List of clusters to merge.
        to : integer
            The id of the new cluster. By default, this is `new_cluster_id()`.

        Returns
        -------

        up : UpdateInfo instance

        """

        if not _is_array_like(cluster_ids):
            raise ValueError("The first argument should be a list or an array.")

        cluster_ids = sorted(cluster_ids)
        if not set(cluster_ids) <= set(self.cluster_ids):
            raise ValueError("Some clusters do not exist.")

        # Find the new cluster number.
        if to is None:
            to = self.new_cluster_id()
        if to < self.new_cluster_id():
            raise ValueError(
                "The new cluster numbers should be higher than {0}.".format(self.new_cluster_id()))

        # NOTE: we could have called self.assign() here, but we don't.
        # We circumvent self.assign() for performance reasons.
        # assign() is a relatively costly operation, whereas merging is a much
        # cheaper operation.

        # Find all spikes in the specified clusters.
        # OPTIM: use spikes_per_cluster rather than going through all spikes.
        spike_ids = np.sort(np.concatenate(
            [_as_array(self._spikes_per_cluster[c]) for c in cluster_ids])).astype(np.int64)
        old_spike_clusters = self._spike_clusters[spike_ids]

        up = self._do_merge(spike_ids, cluster_ids, to)
        undo_state = emit('request_undo_state', self, up)

        # Add to stack.
        # NOTE: spike_ids is shared with spikes_per_cluster[to], so it is not compacted.
        self._undo_stack.add((spike_ids, [to], _compact_ids(old_spike_clusters), undo_state))

        emit('cluster', self, up)
        return up

    def assign(self, spike_ids, spike_clusters_rel=0):
        """Make new spike cluster assignments.

        Parameters
        ----------

        spike_ids : array-like
            List of spike ids.
        spike_clusters_rel : array-like
            Relative cluster ids of the spikes in `spike_ids`. This
            must have the same size as `spike_ids`.

        Returns
        -------

        up : UpdateInfo instance

        Note
        ----

        `spike_clusters_rel` contain *relative* cluster indices. Their values
        don't matter: what matters is whether two give spikes
        should end up in the same cluster or not. Adding a constant number
        to all elements in `spike_clusters_rel` results in exactly the same
        operation.

        The final cluster ids are automatically generated by the `Clustering`
        class. This is because we must ensure that all modified clusters
        get brand new ids. The whole library is based on the assumption that
        cluster ids are unique and "disposable". Changing a cluster always
        results in a new cluster id being assigned.

        If a spike is assigned to a new cluster, then all other spikes
        belonging to the same cluster are assigned to a brand new cluster,
        even if they were not changed explicitely by the `assign()` method.

        In other words, the list of spikes affected by an `assign()` is almost
        always a strict superset of the `spike_ids` parameter. The only case
        where this is not true is when whole clusters change: this is called
        a merge. It is implemented in a separate `merge()` method because it
        is logically much simpler, and faster to execute.

        """

        assert not isinstance(spike_ids, slice)

        # Ensure `spike_clusters_rel` is an array-like.
        if not hasattr(spike_clusters_rel, '__len__'):
            spike_clusters_rel = spike_clusters_rel * np.ones(len(spike_ids), dtype=np.int64)

        spike_ids = _as_array(spike_ids)
        if len(spike_ids) == 0:
            return UpdateInfo()
        assert len(spike_ids) == len(spike_clusters_rel)
        assert spike_ids.min() >= 0
        assert spike_ids.max() < self._n_spikes, "Some spikes don't exist."

        # Normalize the spike-cluster assignment such that
        # there are only new or dead clusters, not modified clusters.
        # This implies that spikes not explicitly selected, but that
        # belong to clusters affected by the operation, will be assigned
        # to brand new clusters.
        spike_ids, cluster_ids = _extend_assignment(
            spike_ids, self._spike_clusters, spike_clusters_rel, self.new_cluster_id(),
            spikes_per_cluster=self._spikes_per_cluster)
        old_spike_clusters = self._spike_clusters[spike_ids]

        up = self._do_assign(spike_ids, cluster_ids)
        undo_state = emit('request_undo_state', self, up)

        # Add the assignment to the undo stack.
        self._undo_stack.add((
            _compact_spike_ids(spike_ids, self._n_spikes), _compact_ids(cluster_ids),
            _compact_ids(old_spike_clusters), undo_state))

        emit('cluster', self, up)
        return up

    def split(self, spike_ids, spike_clusters_rel=0):
        """Split a number of spikes into a new cluster.

        This is equivalent to an `assign()` to a single new cluster.

        Parameters
        ----------

        spike_ids : array-like
            Array of spike ids to split.
        spike_clusters_rel : array-like (or None)
            Array of relative spike clusters.

        Returns
        -------

        up : UpdateInfo instance

        Note
        ----

        The note in the `assign()` method applies here as well. The list
        of spikes affected by the split is almost always a strict superset
        of the spike_ids parameter.

        """
        # self.assign() accepts relative numbers as second argument.
        return self.assign(spike_ids, spike_clusters_rel)

    def undo(self):
        """Undo the last cluster assignment operation.

        Returns
        -------

        up : UpdateInfo instance of the changes done by this operation.

        """
        item = self._undo_stack.back()
        if item is None:
            # Nothing to undo.
            return
        spike_ids, _, old_spike_clusters, undo_state = item

        # Restore the assignment of the spikes affected by the last action. This is O(n_changed)
        # instead of replaying the whole history over the full spike_clusters array.
        up = self._do_assign(_expand_ids(spike_ids), _expand_ids(old_spike_clusters))
        up.history = 'undo'
        # Add the undo_state object from the undone object.
        up.undo_state = undo_state

        emit('cluster', self, up)
        return up

    def redo(self):
        """Redo the last cluster assignment operation.

        Returns
        -------

        up : UpdateInfo instance of the changes done by this operation.

        """
        # Go forward in the stack, and retrieve the new assignment.
        item = self._undo_stack.forward()
        if item is None:
            # No redo has been performed: abort.
            return

        # NOTE: the undo_state object is only returned when undoing.
        # It represents data associated to the state
        # *before* the action. What might be more useful would be the
        # undo_state object of the next item in the list (if it exists).
        spike_ids, cluster_ids, _, undo_state = item
        assert spike_ids is not None

        # We apply the new assignment.
        up = self._do_assign(_expand_ids(spike_ids), _expand_ids(cluster_ids))
        up.history = 'redo'

        emit('cluster', self, up)
        return up
