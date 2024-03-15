"""
Copyright (c) 2023-2024 Blue Brain Project/EPFL
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_pressure(
    graph, params, radii, node_sol, vmin=None, vmax=None, node_label=False, cmap="Blues"
):  # pragma: no cover
    """Plot the pressure on the graph.

    Args:
        graph (vasculatureAPI.PointVasculature): graph containing point vasculature skeleton.
        radii (float): average radius per section.
        params (float): changing parameters.

    Returns:
        float: pressure at each node.
    """
    pos = {}
    for node_id in graph:
        pos[node_id] = list(graph.nodes[node_id]["position"][:2])

    try:
        endfeet_edges = [
            (node_u, node_v)
            for node_u, node_v in graph.edges()
            if graph[node_u][node_v]["endfeet_id"] > -1
        ]
    except BaseException:  # pylint: disable=broad-except
        endfeet_edges = []

    plt.figure(figsize=(10, 5))
    axes = []
    axes.append(plt.gca())

    if "root_id" in params:
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            nodelist=[params["root_id"]],
            node_color="r",
            node_size=50,
            ax=axes[0],
        )

    nx.draw_networkx_edges(
        graph,
        pos=pos,
        edgelist=endfeet_edges,
        width=5 * params["edge_scale"],
        edge_color="r",
        alpha=1,
    )

    nx.draw_networkx_edges(
        graph,
        pos=pos,
        width=params["edge_scale"] * radii / np.max(radii),
        edge_color="k",
    )

    nodes = nx.draw_networkx_nodes(
        graph,
        pos=pos,
        node_size=params["node_scale"],
        # * (node_sol - np.log10(params['p_min'])),
        node_color=node_sol,
        ax=axes[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if node_label:
        nx.draw_networkx_labels(graph, pos=pos)

    plt.colorbar(nodes, ax=axes[0])  # , label='node pressure (log scale)')

    axes[0].axis("off")


def plot_resistance(
    graph,
    params,
    radii,
    edge_vmin=None,
    edge_vmax=None,
    edge_label=False,
    edge_cmap=plt.cm.twilight,
):  # pragma: no cover
    """Plot the resistance on the graph.

    Args:
        graph (vasculatureAPI.PointVasculature): graph containing point vasculature skeleton.
        radii (float): average radius per section.
        params (float): changing parameters.
        node_sol

    Returns:
        float: resistance at each edge.
    """
    pos = {}
    for node_id in graph:
        pos[node_id] = list(graph.nodes[node_id]["position"][:2])

    try:
        endfeet_edges = [
            (node_u, node_v)
            for node_u, node_v in graph.edges()
            if graph[node_u][node_v]["endfeet_id"] > -1
        ]
    except BaseException:  # pylint: disable=broad-except
        endfeet_edges = []

    plt.figure(figsize=(10, 5))
    axes = []
    axes.append(plt.gca())

    if "root_id" in params:
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            nodelist=[params["root_id"]],
            node_color="r",
            node_size=50,
            ax=axes[0],
        )

    nx.draw_networkx_edges(
        graph,
        pos=pos,
        edgelist=endfeet_edges,
        width=5 * params["edge_scale"],
        edge_color="r",
        alpha=1,
    )

    nx.draw_networkx_edges(
        graph,
        pos=pos,
        width=params["edge_scale"] * radii / np.max(radii),
        # width = 1*np.array(radii),
        edge_color=params["resistances"],
        ax=axes[0],
        edge_cmap=edge_cmap,
        # edge_vmin=edge_vmin,
        # edge_vmax=edge_vmax,
    )

    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        node_size=params["node_scale"],
        node_color="k",
    )

    if edge_label:
        nx.draw_networkx_labels(graph, pos=pos)

    plt.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(edge_vmin, edge_vmax), cmap=edge_cmap),
        ax=axes[0],
    )

    axes[0].axis("off")


def plot_on_graph(nx_graph, graph, pos, node_key="pressure", edge_key="flow"):
    """Plot data on a graph."""
    input_nodes = np.argwhere(graph.degrees == 1).T[0]
    graph.node_properties.loc[input_nodes, node_key] = np.nan
    mask = graph.edge_properties[graph.edge_properties.end_node.isin(input_nodes)].index
    graph.edge_properties.loc[mask, edge_key] = np.nan
    if node_key is not None:
        nx.draw_networkx_nodes(
            nx_graph,
            pos=pos,
            node_size=2,
            node_color=graph.node_properties[node_key].to_numpy(),
            cmap=plt.get_cmap("Reds"),
        )
    if edge_key is not None:
        nx.draw_networkx_nodes(nx_graph, pos=pos, nodelist=input_nodes, node_size=5, node_color="g")
    e = nx.draw_networkx_edges(
        nx_graph,
        pos=pos,
        edge_color=graph.edge_properties[edge_key].to_numpy(),
        edge_cmap=plt.get_cmap("Blues"),
    )
    plt.colorbar(e, shrink=0.5)
    edgelist = [e for i, e in enumerate(nx_graph.edges) if i in mask]
    nx.draw_networkx_edges(nx_graph, pos=pos, edgelist=edgelist, edge_color="g")
    plt.axis("equal")
