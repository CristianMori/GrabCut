"""This module contains the definition for a graph object that will be used to calculate min flow."""
import numpy as np
from enum import IntEnum


class TerminationType(IntEnum):
    """Represents the termination type."""

    source = 0
    sink = 1


class Node:
    """Definition of a node object."""

    def __init__(self):
        """Initialize a node."""
        self.first = None
        self.parent = None
        self.next = None
        self.ts = 0
        self.dist = 0
        self.is_sink = True
        self.is_marked = True
        self.is_in_changed_list = True
        self.tr_cap = 0


class Arc:
    """Definition of an arc object."""

    def __init__(self):
        """Initialize an arc."""
        self.node = None
        self.next = None
        self.sister = None
        self.r_cap = 0


class LinkedListPointer:
    """Definition of a linked list pointer."""

    def __init__(self):
        """Initialize a pointer."""
        self.ptr = None
        self.next = None


class Graph:
    """Definition of a graph object."""

    def __init__(self):
        """Initialize with the specified size."""
        self.reset()
        self.queue_first = [None, None]
        self.queue_last = [None, None]
        self.orphan_first = None
        self.orphan_last = None
        self.changed_list = []
        self.terminal_parent = 97
        self.orphan_parent = 100

    def reset(self):
        """Reset the graph."""
        self.next_id = 0
        self.nodes = []
        self.arcs = []
        self.flow = 0
        self.maxflow_iteration = 0

    def add_nodes(self, num_nodes):
        """Add num_nodes nodes."""
        first_id = self.next_id
        self.next_id += num_nodes
        for i in range(num_nodes):
            self.nodes.append(Node())
        return list(range(first_id, self.next_id))

    def add_tweights(self, node_id, cap_source, cap_sink):
        """Add terminal weights from node id."""
        delta = self.nodes[node_id].tr_cap
        if delta > 0:
            cap_source += delta
        else:
            cap_source -= delta
        self.flow += min(cap_source, cap_sink)
        self.nodes[node_id].tr_cap = cap_source - cap_sink

    def add_edge(self, i, j, cap, rev_cap):
        """Add an edge between nodes i and j."""
        node_i = self.nodes[i]
        node_j = self.nodes[j]

        a = Arc()
        a_rev = Arc()
        a.sister = a_rev
        a_rev.sister = a
        a.next = node_i.first
        node_i.first = a
        a_rev.next = node_j.first
        node_j.first = a_rev
        a.head = node_j
        a_rev.head = node_i
        a.r_cap = cap
        a_rev.r_cap = rev_cap
        self.arcs.append(a)
        self.arcs.append(a_rev)

    def remove_from_changed_list(self, node_id):
        """Remove from the changed list."""
        self.nodes[node_id].is_in_changed_list = False

    def what_segment(self, node_id, default_segm=TerminationType.source):
        """Check the segment."""
        node = self.nodes[node_id]
        if node.parent:
            return TerminationType.sink if node.is_sink else TerminationType.source
        else:
            return default_segm

    def mark_node(self, node_id):
        """Mark the specified node."""
        node = self.nodes[node_id]
        if not node.next:
            if self.queue_last[1]:
                self.queue_last[1].next = node
            else:
                self.queue_first[1] = node
            self.queue_last[1] = node
            node.next = node
        node.is_marked = True

    def set_node_active(self, node):
        """Set a node active."""
        if not node.next:
            if self.queue_last[1]:
                self.queue_last[1].next = node
            else:
                self.queue_first[1] = node
            self.queue_last[1] = node
            node.next = node

    def get_next_active_node(self):
        """Get the next active node."""
        while True:
            node = self.queue_first[0]
            if not node:
                self.queue_first[0] = self.queue_first[1]
                node = self.queue_first[1]
                self.queue_last[0] = self.queue_last[1]
                self.queue_first[1] = None
                self.queue_last[1] = None
                if not node:
                    return None
            if node.next == node:
                self.queue_first[0], self.queue_last[0] = None, None
            else:
                self.queue_first[0] = node.next
            node.next = None
            if node.parent:
                return node

    def set_orphan_front(self, node):
        """Add the orphan to the front of the orphans list."""
        node.parent = self.orphan_parent
        np = LinkedListPointer()
        np.ptr = node
        np.next = self.orphan_first
        self.orphan_first = np

    def set_orphan_back(self, node):
        """Add the orphan to the back of the orphans list."""
        node.parent = self.orphan_parent
        np = LinkedListPointer()
        np.ptr = node
        if self.orphan_last:
            self.orphan_last.next = np
        else:
            self.orphan_first = np
        self.orphan_last = np
        np.next = None

    def add_to_changed_list(self, node):
        """Add the node to the changed list."""
        if self.changed_list and not node.is_in_changed_list:
            self.changed_list.append(node)
            node.in_changed_list = True

    def maxflow_init(self):
        """Initialize for maxflow."""
        self.queue_first = [None, None]
        self.queue_last = [None, None]
        self.orphan_first = None
        self.changed_list = []

        self.time = 0
        for node in self.nodes:
            node.next = None
            node.is_marked = False
            node.is_in_changed_list = False
            node.ts = self.time
            if node.tr_cap > 0:
                node.is_sink = False
                node.parent = self.terminal_parent
                self.set_node_active(node)
                node.dist = 1
            elif node.tr_cap < 0:
                node.is_sink = True
                node.parent = self.terminal_parent
                self.set_node_active(node)
                node.dist = 1
            else:
                node.parent = None

    def maxflow_reuse_trees_init(self):
        """Initialize the trees to be reused."""
        nq = self.queue_first[1]
        self.queue_first = [None, None]
        self.queue_last = [None, None]
        self.orphan_first = None
        self.orphan_last = None
        i = nq
        self.time += 1
        while i:
            nq = i.next
            if nq == i:
                nq = None
            i.next = None
            i.is_marked = False
            self.set_node_active(i)

            if i.tr_cap == 0:
                if i.parent:
                    self.set_orphan_back(i)
                    i = nq
                    continue
            if i.tr_cap > 0:
                if (not i.parent) or i.is_sink:
                    i.is_sink = False
                    a = i.first
                    while a:
                        j = a.head
                        if not j.is_marked:
                            if j.parent == a.sister:
                                self.set_orphan_back(j)
                            if j.parent and j.is_sink and a.r_cap > 0:
                                self.set_node_active(j)
                        a = a.next
                    self.add_to_changed_list(i)
            else:
                if (not i.parent) or (not i.is_sink):
                    i.is_sink = True
                    a = i.first
                    while a:
                        j = a.head
                        if not j.is_marked:
                            if j.parent == a.sister:
                                self.set_orphan_back(j)
                            if j.parent and (not j.is_sink) and a.sister.r_cap > 0:
                                self.set_node_active(j)
                        a = a.next
                    self.add_to_changed_list(i)
            i.parent = self.terminal_parent
            i.ts = self.time
            i.dist = 1
            i = nq

        np = self.orphan_first
        while np:
            self.orphan_first = np.next
            i = np.ptr
            if not self.orphan_first:
                self.orphan_last = None
            if i.is_sink:
                self.process_sink_orphan(i)
            else:
                self.process_source_orphan(i)
            np = self.orphan_first

    def augment(self, middle_arc):
        """Perform the augmentation step."""
        bottleneck = middle_arc.r_cap

        # Step 1: Find the bottleneck
        i = middle_arc.sister.head
        while True:
            a = i.parent
            if a == self.terminal_parent:
                break
            if bottleneck > a.sister.r_cap:
                bottleneck = a.sister.r_cap
            i = a.head
        if bottleneck > i.tr_cap:
            bottleneck = i.tr_cap

        i = middle_arc.head
        while True:
            a = i.parent
            if a == self.terminal_parent:
                break
            if bottleneck > a.r_cap:
                bottleneck = a.r_cap
            i = a.head
        if bottleneck > -i.tr_cap:
            bottleneck = -i.tr_cap

        # Step 2: Augment
        middle_arc.sister.r_cap += bottleneck
        middle_arc.r_cap -= bottleneck

        i = middle_arc.sister.head
        while True:
            a = i.parent
            if a == self.terminal_parent:
                break
            a.r_cap += bottleneck
            a.sister.r_cap -= bottleneck
            if not a.sister.r_cap:
                self.set_orphan_front(i)
            i = a.head

        i.tr_cap -= bottleneck
        if not i.tr_cap:
            self.set_orphan_front(i)

        i = middle_arc.head
        while True:
            a = i.parent
            if a == self.terminal_parent:
                break
            a.sister.r_cap += bottleneck
            a.r_cap -= bottleneck
            if not a.r_cap:
                self.set_orphan_front(i)
            i = a.head

        i.tr_cap += bottleneck
        if not i.tr_cap:
            self.set_orphan_front(i)

        self.flow += bottleneck

    def process_source_orphan(self, node):
        """Process the source orphan."""
        a0_min = None
        d_min = float('inf')

        a0 = node.first
        while a0:
            if a0.sister.r_cap:
                j = a0.head
                a = j.parent
                if not j.is_sink and a:
                    d = 0
                    while True:
                        if j.ts == self.time:
                            d += j.dist
                            break
                        a = j.parent
                        d += 1
                        if a == self.terminal_parent:
                            j.ts = self.time
                            j.dist = 1
                            break
                        if a == self.orphan_parent:
                            d = float('inf')
                            break
                        j = a.head

                    if d < float('inf'):
                        if d < d_min:
                            a0_min = a0
                            d_min = d
                        j = a0.head
                        while j.ts != self.time:
                            j.ts = self.time
                            j.dist = d
                            d -= 1
                            j = j.parent.head
            a0 = a0.next

        node.parent = a0_min
        if node.parent:
            node.ts = self.time
            node.dist = d_min + 1
        else:
            self.add_to_changed_list(node)
            a0 = node.first
            while a0:
                j = a0.head
                a = j.parent
                if not j.is_sink and a:
                    if a0.sister.r_cap:
                        self.set_node_active(j)
                    if a != self.terminal_parent and a != self.orphan_parent and a.head == node:
                        self.set_orphan_back(j)
                a0 = a0.next

    def process_sink_orphan(self, node):
        """Process the sink orphan."""
        a0_min = None
        d_min = float('inf')

        a0 = node.first
        while a0:
            if a0.r_cap:
                j = a0.head
                a = j.parent
                if j.is_sink and a:
                    d = 0
                    while True:
                        if j.ts == self.time:
                            d += j.dist
                            break
                        a = j.parent
                        d += 1
                        if a == self.terminal_parent:
                            j.ts = self.time
                            j.dist = 1
                            break
                        if a == self.orphan_parent:
                            d = float('inf')
                            break
                        j = a.head

                    if d < float('inf'):
                        if d < d_min:
                            a0_min = a0
                            d_min = d
                        j = a0.head
                        while j.ts != self.time:
                            j.ts = self.time
                            j.dist = d
                            d -= 1
                            j = j.parent.head
            a0 = a0.next

        node.parent = a0_min
        if node.parent:
            node.ts = self.time
            node.dist = d_min + 1
        else:
            self.add_to_changed_list(node)
            a0 = node.first
            while a0:
                j = a0.head
                a = j.parent
                if j.is_sink and a:
                    if a0.r_cap:
                        self.set_node_active(j)
                    if a != self.terminal_parent and a != self.orphan_parent and a.head == node:
                        self.set_orphan_back(j)
                a0 = a0.next

    def maxflow(self, reuse_trees=False, changed_list=[]):
        """Perform maxflow."""
        curr_node = None
        self.changed_list = changed_list
        if self.maxflow_iteration == 0 and reuse_trees:
            print("Reuse trees cannot be used in the first iteration")
            return
        if changed_list and not reuse_trees:
            print("Changed list cannot be used without reuse_trees")
            return

        if reuse_trees:
            self.maxflow_reuse_trees_init()
        else:
            self.maxflow_init()
        print("Finished maxflow initialization")

        while True:
            i = curr_node
            if i:
                i.next = None
                if not i.parent:
                    i = None
            if not i:
                i = self.get_next_active_node()
                if not i:
                    break

            # growth
            if not i.is_sink:
                # grow source tree
                a = i.first
                while a:
                    if a.r_cap:
                        j = a.head
                        if not j.parent:
                            j.is_sink = False
                            j.parent = a.sister
                            j.ts = i.ts
                            j.dist = i.dist + 1
                            self.set_node_active(j)
                            self.add_to_changed_list(j)
                        elif j.is_sink:
                            break
                        elif j.ts <= i.ts and j.dist > i.dist:
                            j.parent = a.sister
                            j.ts = i.ts
                            j.dist = i.dist + 1
                    a = a.next
            else:
                # grow sink tree
                a = i.first
                while a:
                    if a.sister.r_cap:
                        j = a.head
                        if not j.parent:
                            j.is_sink = True
                            j.parent = a.sister
                            j.ts = i.ts
                            j.dist = i.dist + 1
                            self.set_node_active(j)
                            self.add_to_changed_list(j)
                        elif not j.is_sink:
                            a = a.sister
                            break
                        elif j.ts <= i.ts and j.dist > i.dist:
                            j.parent = a.sister
                            j.ts = i.ts
                            j.dist = i.dist + 1
                    a = a.next

            self.time += 1
            if a:
                i.next = i
                curr_node = i

                # augmentation
                self.augment(a)

                # adoption
                np = self.orphan_first
                while np:
                    np_next = np.next
                    np.next = None
                    np = self.orphan_first
                    while np:
                        self.orphan_first = np.next
                        i = np.ptr
                        if not self.orphan_first:
                            self.orphan_last = None
                        if i.is_sink:
                            self.process_sink_orphan(i)
                        else:
                            self.process_source_orphan(i)
                    self.orphan_first = np_next
                    np = self.orphan_first
            else:
                curr_node = None

        self.maxflow_iteration += 1
        return self.flow

    # below this point are helper functions used in cut.py
    def add_grid_nodes(self, shape):
        """Add a set of nodes."""
        num_nodes = np.prod(shape)
        first_id = self.add_nodes(num_nodes)[0]
        nodes = np.arange(first_id, first_id + num_nodes)
        return np.reshape(nodes, shape)

    def add_tedge(self, i, cap_source, cap_sink):
        """Wrapper for add_tweights."""
        self.add_tweights(i, cap_source, cap_sink)

    def get_grid_segments(self, nodeids):
        """Get the grid segments."""
        sgm = np.zeros(nodeids.shape, dtype=np.bool)
        for (x, y), ind in np.ndenumerate(nodeids):
            sgm[x][y] = (self.what_segment(ind) == TerminationType.sink)
        return sgm

    # def add_grid_edges(self, nodeids, weights=1, structure=None, symmetric=False):
    #     """Add a grid of edges."""
    #     if not structure:
    #         structure = np.array([[0, 1, 0],
    #                               [1, 0, 1],
    #                               [0, 1, 0]])
    #     ndim = nodeids.ndim
    #     if type(weights) != np.ndarray:
    #         weights = np.full(structure.shape, weights)

    # def fill_all_weights(self, nodeids, diag_left, diag_right, up, left):
    #     """Fill all weights using the specified weights."""
    #     for y, x in zip(range(nodeids.shape[0]), range(nodeids.shape[1])):
    #         node_id = nodeids[y][x]
    #         if y > 0 and x > 0:

    #             diag_left[y][x] = 50 / np.sqrt(2) * np.exp(-beta * (z_m - self.img[y - 1][x - 1])**2)
    #         if y > 0 and x < self.width - 1:
    #             diag_right[y][x] = 50 / np.sqrt(2) * np.exp(-beta * (z_m - self.img[y - 1][x + 1])**2)
    #         if x > 0:
    #             left[y][x] = 50 * np.exp(-beta * (z_m - self.img[y][x - 1])**2)
    #         if y > 0:
    #             up[y][x] = 50 * np.exp(-beta * (z_m - self.img[y - 1][x])**2)
