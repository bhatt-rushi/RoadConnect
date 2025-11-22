# /src/model/base.py
from typing import List, Tuple
from utils import config
from model import graph
import networkx as nx
from model import data
from tqdm import tqdm

class Model:
    def __init__(self) -> None:
        # TODO: Check that all CRS match
        self.load_config_values()
        self.generate_base_graph()

        self.run()
        pass

    def load_config_values(self):
        # Load values from configuration file
        self.rainfall_events: List[float] = config.get_rainfall_values()

    def generate_base_graph(self):
        # Generate base graph
        self.base_graph = graph.Graph()
        self.base_graph.add_nodes(data.drains.get_nodes())
        self.base_graph.add_nodes(data.ponds.get_nodes())

    def run(self) -> List[Tuple[float, nx.DiGraph]]:
        results = []

        for rainfall_event_total in tqdm(self.rainfall_events, total=len(self.rainfall_events)):
            g = self.base_graph.copy()
            g.simulate_rainfall(rainfall_event_total)
            g.print()
            results.append((rainfall_event_total, g.to_networkx_graph()))
        return results
