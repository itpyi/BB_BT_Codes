"""Bipartite edge-coloring used for CNOT layer scheduling.

Produces an optimal edge coloring (class-1) following the constructive proof
for bipartite graphs. Each color class is a set of disjoint edges that can be
executed in parallel as one CNOT layer.
"""

import networkx as nx
import numpy as np
import stim
from collections import namedtuple
from typing import List, Set, Tuple, Any, Iterable, Union, Any as _Any, Sequence, cast
from scipy.sparse import csr_matrix
from networkx.algorithms import bipartite
from networkx import relabel_nodes


def _canonicalize_edge(e: tuple) -> tuple:
    """Return an order-invariant tuple key for an edge (u, v [, key])."""
    (u, v) = e[:2]  # [:2] in case the graph is weighted
    (u, v) = (u, v) if u < v else (v, u)
    return (u, v) + e[2:]


def _edges_with_keys(G: nx.Graph) -> _Any:
    """Return an edge view (type-erased) for compatibility with mypy.

    NetworkX edge views are not precisely typed; we erase the type to _Any and
    document usage via EdgeTuple in the public API.
    """
    try:
        return G.edges(keys=True)
    except TypeError:
        return G.edges()


EdgeTuple = Union[Tuple[int, int], Tuple[int, int, Any]]


def edge_color_bipartite(bipartite_graph: nx.Graph) -> List[Set[EdgeTuple]]:
    """Return an optimal edge coloring in O(|V||E|) for a bipartite graph.

    Returns a list of color classes. Each class is a set of edges represented
    by canonicalized tuples.
    """

    G = bipartite_graph.to_undirected()

    if not nx.is_bipartite(G):
        raise RuntimeError("Graph must be bipartite")

    if nx.number_of_selfloops(G) > 0:
        raise RuntimeError("Graph must not contain self loops")

    graph_degree = max(map(lambda x: x[1], G.degree()))

    ColorSet = namedtuple("ColorSet", ["vertices", "edges"])
    colorings = [ColorSet(set(), set()) for _ in range(graph_degree)]

    for edge in _edges_with_keys(G):
        (u, v) = edge[:2]
        u_set = None
        try:
            # Prefer a color where neither endpoint is used.
            u_set = next(
                x for x in colorings if u not in x.vertices and v not in x.vertices
            )
        except StopIteration:
            # Otherwise, fix two partial colorings by alternating along a chain
            # so we can add (u, v) to one of them, preserving disjointness.
            u_set = next(x for x in colorings if u not in x.vertices)
            v_set = next(x for x in colorings if v not in x.vertices)

            # Compute an edge 2-coloring in the subgraph v_set ∪ u_set and swap
            # colors along the alternating path to free both endpoints in u_set.
            def filter_edge(u, v, key=None):
                edge = (u, v) if key is None else (u, v, key)
                return (
                    _canonicalize_edge(edge) in u_set.edges
                    or _canonicalize_edge(edge) in v_set.edges
                )

            uv_subgraph = nx.subgraph_view(
                G,
                filter_node=lambda x: (x in u_set.vertices or x in v_set.vertices),
                filter_edge=filter_edge,
            )

            # Follow chain starting at v; flip class membership along the chain.
            u_to_v_set = set()
            v_to_u_set = set()
            for uv_edge in nx.edge_dfs(uv_subgraph, v):
                if _canonicalize_edge(uv_edge) in u_set.edges:
                    u_to_v_set.add(_canonicalize_edge(uv_edge))
                else:
                    v_to_u_set.add(_canonicalize_edge(uv_edge))

            u_to_v_vertices = set(v for edges in u_to_v_set for v in edges)
            v_to_u_vertices = set(v for edges in v_to_u_set for v in edges)

            u_set.edges.difference_update(u_to_v_set)
            u_set.vertices.difference_update(u_to_v_vertices)
            u_set.edges.update(v_to_u_set)
            u_set.vertices.update(v_to_u_vertices)

            v_set.edges.difference_update(v_to_u_set)
            v_set.vertices.difference_update(v_to_u_vertices)
            v_set.edges.update(u_to_v_set)
            v_set.vertices.update(u_to_v_vertices)

        # Add the original edge to the chosen color class.
        u_set.vertices.add(u)
        u_set.vertices.add(v)
        u_set.edges.add(_canonicalize_edge(edge))

    return [v.edges for v in colorings]


def generate_edge_colored_syndrome_circuit(
    H: Any,
    checks: Sequence[int],
    stab_type: int,
    p1: float,
    p2: float,
    seed: int,
) -> stim.Circuit:
    """Return Stim circuit for one stabilizer extraction layer using edge-coloring.

    Parameters
    - H: biadjacency matrix (CSR/array) mapping check-to-data for a CSS half.
    - checks: iterable of qubit indices for the ancilla (check) qubits.
    - stab_type: 0 for Z-stabilizers (no Hadamards), 1 for X-stabilizers
      (surround CNOTs with H on checks to rotate basis).
    - p1, p2: single/two-qubit depolarizing error rates applied after gates.
    - seed: RNG seed to shuffle color layers (0 means no shuffle).
    """
    # Ensure SciPy CSR for NetworkX biadjacency conversion
    H_csr = csr_matrix(H)
    m, n = H_csr.shape

    # No Idle error
    # p_idle = p1 / 100

    # Build Tanner graph and relabel to match the global qubit index map used
    # by the full circuit (data first, then X-checks, then Z-checks).
    tanner_graph = bipartite.from_biadjacency_matrix(H_csr)
    mapping = {i: checks[i] for i in range(m)}
    mapping.update({i: i - m for i in range(m, n + m)})
    tanner_graph = relabel_nodes(tanner_graph, mapping)

    # Edge-color the bipartite graph to schedule disjoint CNOT layers.
    coloring = edge_color_bipartite(tanner_graph)
    if seed != 0:
        rng = np.random.default_rng(seed=seed)
        # Shuffle in-place; cast to Any to satisfy mypy on numpy API.
        rng.shuffle(cast(Any, coloring), axis=0)

    c = stim.Circuit()

    # For X-stabilizers, rotate ancillas into X basis via H.
    c.append("TICK")
    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p1)
        c.append("TICK")

    for r in coloring:
        # Apply each edge-color class as one CNOT layer.
        data_qbts = set(np.arange(n))
        for g in r:
            # g = (data_idx, check_idx), consume data from the idle set.
            data_qbts.remove(g[0])
            targets = g[::-1] if stab_type else g
            c.append("CX", targets)
            c.append("DEPOLARIZE2", targets, p2)
        # Idle data qubits accrue single-qubit depolarizing noise this layer.
        # 2025/9/3 Comment idle error
        # c.append("DEPOLARIZE1", data_qbts, p_idle)
        c.append("TICK")

    # Undo the basis rotation for X-stabilizers.
    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p1)
        c.append("TICK")
    return c
