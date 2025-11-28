# /src/model/base.py
from typing import List, Tuple
from utils import config
from model import graph, visualization
import networkx as nx
from tqdm import tqdm

class Model:
    def __init__(self) -> None:
        # TODO: Check that all CRS match
        self.load_config_values()
        self.base_graph = graph.Graph()

        self.__post_init__()

    def __post_init__(self) -> None:
        self.results : List[Tuple[float, nx.DiGraph]] = []
        self.run()

    def load_config_values(self):
        # Load values from configuration file
        self.rainfall_events: List[float] = config.get_rainfall_values()

    def run(self):
        # For each rainfall event, get a graph
        for rainfall_event_total in tqdm(self.rainfall_events, total=len(self.rainfall_events)):
            g = self.base_graph.copy()
            g.simulate_rainfall(rainfall_event_total)

            nx_graph = g.to_networkx_graph()
            for point in nx_graph.nodes:
                nodedata = nx_graph.nodes[point]['nodedata']
                nodedata.visualization = visualization.visualization_information(graph=nx_graph, node=nodedata)

            self.results.append((rainfall_event_total, nx_graph))
