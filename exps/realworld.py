import json
import random

import networkx as nx
import re
import sammo.utils
from orjson import orjson

from exps.base import Experiment
from exps.synthetic import ReachabilityExperiment


class MostNegativeReviewExperiment(Experiment):
    METRIC = "accuracy"
    LOADED_DATASET = None
    OUTPUT_SCHEMA = {"most_negative": 42}
    SETTINGS = {
        "list_length": {"default": 6, "sweep": [6, 10, 20, 50, 80, 100]},
        "task": {"default": "majority_review", "sweep": ["majority_review"]},
    }
    XAXIS = "list_length"
    OTHER_AXES = ("task", "n_runs")

    def generate_data(self, model_shorthand=None):
        rng = random.Random(int(self._rng.integers(0, 2**16 - 1)))

        n = self.settings["list_length"]
        if self.__class__.LOADED_DATASET is None:
            print("Loading dataset...")
            self.__class__.LOADED_DATASET = orjson.loads(
                (sammo.utils.MAIN_PATH.parent / "processed_data" / "space.json").read_bytes()
            )

        n_data = len(self.LOADED_DATASET)
        data, outputs = list(), list()
        for i in range(self.settings["n_runs"]):
            pos_reviews = rng.sample(self.LOADED_DATASET[i % n_data]["1"], k=n - 1)
            neg_reviews = rng.sample(self.LOADED_DATASET[i % n_data]["-1"], k=1)
            all_reviews = [{"label": "pos", "text": t} for t in pos_reviews] + [
                {"label": "neg", "text": t} for t in neg_reviews
            ]
            rng.shuffle(all_reviews)
            data.append(all_reviews)
            outputs += [i for i, v in enumerate(all_reviews) if v["label"] == "neg"]

        instruction = "Return the id of the most negative review."
        inputs = [
            {
                "data": d,
                "rendered_data": json.dumps(
                    [{"id": i, "review": d["text"]} for i, d in enumerate(d)], indent=2, ensure_ascii=False
                ),
                "setting": self.settings,
                "instruction": instruction,
            }
            for d in data
        ]

        return inputs, outputs


class MajorityReviewExperiment(Experiment):
    METRIC = "accuracy"
    LOADED_DATASET = None
    OUTPUT_SCHEMA = {"majority_is_positive": True}
    SETTINGS = {
        "list_length": {"default": 6, "sweep": [6, 10, 20, 50, 80, 100]},
        "task": {"default": "majority_review", "sweep": ["majority_review"]},
        "slack": {"default": 3, "sweep": [3]},
    }
    XAXIS = "list_length"
    OTHER_AXES = ("task", "n_runs")

    def generate_data(self, model_shorthand=None):
        rng = random.Random(int(self._rng.integers(0, 2**16 - 1)))

        n = self.settings["list_length"]
        if self.__class__.LOADED_DATASET is None:
            print("Loading dataset...")
            self.__class__.LOADED_DATASET = orjson.loads(
                (sammo.utils.MAIN_PATH.parent / "processed_data" / "space.json").read_bytes()
            )

        n_data = len(self.LOADED_DATASET)
        data, outputs = list(), list()
        slack = self.settings["slack"]
        for output in [True, False]:
            for i in range(self.settings["n_runs"] // 2):
                pos_reviews = rng.sample(self.LOADED_DATASET[i % n_data]["1"], k=n // 2 + slack * int(output))
                neg_reviews = rng.sample(self.LOADED_DATASET[i % n_data]["-1"], k=n // 2 + slack * int(not output))
                all_reviews = pos_reviews + neg_reviews
                rng.shuffle(all_reviews)
                data.append(all_reviews)
                outputs.append(output)

        instruction = "Output true if the majority of the following reviews is positive, else false."
        inputs = [
            {
                "data": d,
                "rendered_data": json.dumps(
                    [{"id": i, "review": d} for i, d in enumerate(d)], indent=2, ensure_ascii=False
                ),
                "setting": self.settings,
                "instruction": instruction,
            }
            for d in data
        ]

        return inputs, outputs


class VariableTrackingExperiment(ReachabilityExperiment):
    METRIC = "accuracy"
    OUTPUT_SCHEMA = {"is_equal": True}
    SETTINGS = {
        "task": {"default": "tracking", "sweep": ["tracking"]},
        "n_paths": {"default": 2, "sweep": [2]},
        "list_length": {"default": 6, "sweep": [6, 10, 20, 50, 80, 100]},
    }
    XAXIS = "list_length"
    OTHER_AXES = ("n_paths", "task")

    @classmethod
    def _graph_to_text(cls, edges, starting_nodes, s, t, mapping):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        values = {n: alphabet[i] for i, n in enumerate(starting_nodes)}
        output = f'In the Python code below, is x{mapping[t]} == "{values[s]}" at the end of execution?\n'
        output += "```python\n"

        for n in starting_nodes:
            output += f'x{mapping[n]} = "{values[n]}"\n'
        for i, j in edges:
            output += f"x{mapping[j]} = x{mapping[i]}\n"
        output += "```\n"

        return output

    def _oracle(cls, data):
        # extract everything in the code block
        code_block = re.search(r"```python\n(.*?)\n```", data, re.DOTALL).group(1)
        # extract the variable names
        a, b = re.findall(r'x(\d+) == "(\w+)"', data)[0]
        test_code = code_block + f"\nresult = (x{a} == '{b}')"
        states = {}
        exec(test_code, dict(), states)
        return states["result"]

    def generate_path_graph(self, connected):
        n_paths = self.settings["n_paths"]
        path_length = (self.settings["list_length"] // self.settings["n_paths"]) - 1

        paths = [[(f"r{r}c{c}", f"r{r}c{c+1}") for c in range(path_length)] for r in range(n_paths)]
        nodes = [f"r{r}c{c}" for c in range(path_length + 1) for r in range(n_paths)]

        # sample node from earlier
        j, i = sorted(self._rng.choice(path_length, size=2, replace=False))  # generate j < i
        crossover = self._rng.permutation(n_paths)

        for p, l in enumerate(crossover):
            paths[p][i] = (f"r{l}c{j}", f"r{p}c{i+1}")
        s_id = self._rng.choice(n_paths)
        connects_to = crossover[s_id]
        if not connected:
            t_id = self._rng.choice([v for v in crossover if v != connects_to])
        else:
            t_id = connects_to
        s, t = f"r{s_id}c0", f"r{t_id}c{path_length}"

        # choose a random topological ordering of the nodes
        graph = nx.DiGraph(sum(paths, []))
        node_ordering = list()
        starting_nodes = None
        for gen in nx.topological_generations(graph):
            if starting_nodes is None:
                starting_nodes = list(gen)
            node_ordering.extend(self._rng.permutation(sorted(gen)))

        # for each node, add incoming edges in random order
        edges = []
        for node in node_ordering:
            predecessors = list(graph.predecessors(node))
            if predecessors:
                edges.extend(self._rng.permutation([(predecessor, node) for predecessor in predecessors]).tolist())

        mapping = {node: i for i, node in zip(self._rng.permutation(len(nodes)), nodes)}
        return {
            "data": "",
            "rendered_data": self._graph_to_text(edges, starting_nodes, s, t, mapping),
            "setting": self.settings,
            "instruction": f"",
        }, connected
