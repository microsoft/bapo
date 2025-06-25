import json
import re

import networkx as nx
import numpy as np
import scipy.stats

from exps.base import Experiment


class PercentileExperiment(Experiment):
    METRIC = "accuracy"
    OUTPUT_SCHEMA = {"computed_percentile": -1}
    SETTINGS = {
        "distribution": {"default": "uniform", "sweep": ["lognormal"]},
        "percentile": {"default": 25, "sweep": [100]},
        "use_percentiles_aliases": {"default": True},
    }
    PERCENTILE_ALIASES = {0: "minimum", 50: "median", 100: "maximum"}
    XAXIS = "list_length"
    OTHER_AXES = ("percentile", "distribution", "use_percentiles_aliases")

    def truncated_lognormal(self, lower=0, upper=20, loc=0, scale=1, size=10000):
        # Transform parameters for truncation
        l_prime = (np.log(lower + 1) - loc) / scale
        u_prime = (np.log(upper + 1) - loc) / scale

        sample = scipy.stats.truncnorm(l_prime, u_prime, loc=loc, scale=scale).rvs(size=size, random_state=self._rng)
        return np.round(np.exp(sample) - 1).astype(int)

    def _draw_samples(self):
        list_length = self.settings["list_length"] + 1
        n_runs = self.settings["n_runs"]
        rng = self._rng
        if self.settings["distribution"] == "uniform":
            return rng.integers(0, 1000, size=(n_runs, list_length)).tolist()
        elif self.settings["distribution"] == "lognormal":
            return self.truncated_lognormal(0, 1000, scale=10, size=(n_runs, list_length)).tolist()
        else:
            return rng.geometric(0.1, size=(n_runs, list_length)).tolist()

    def generate_data(self, model_shorthand=None):
        def isinteger(x):
            return np.equal(np.mod(x, 1), 0)

        data = self._draw_samples()
        percentile = self.settings["percentile"]

        # check lengths
        assert all(len(d) == self.settings["list_length"] + 1 for d in data)
        assert all(len(d) % 2 == 1 for d in data)

        if self.settings["use_percentiles_aliases"] and percentile in self.PERCENTILE_ALIASES:
            prompt = self.PERCENTILE_ALIASES[percentile]
        else:
            prompt = f"{percentile}-th percentile"

        inputs = [
            {
                "data": d,
                "rendered_data": json.dumps(d),
                "setting": self.settings,
                "instruction": f"Output the {prompt} of this list:",
            }
            for d in data
        ]
        outputs = [np.percentile(d, percentile) for d in data]
        assert all([isinteger(x) for x in outputs])
        return inputs, outputs


class MinExperiment(PercentileExperiment):
    SETTINGS = PercentileExperiment.SETTINGS.copy()
    SETTINGS["percentile"] = {"default": 0}
    OUTPUT_SCHEMA = {"computed_min": -1}


class MaxExperiment(PercentileExperiment):
    SETTINGS = PercentileExperiment.SETTINGS.copy()
    SETTINGS["percentile"] = {"default": 100}
    OUTPUT_SCHEMA = {"computed_max": -1}


class MajorityExperiment(Experiment):
    METRIC = "accuracy"
    OUTPUT_SCHEMA = {"majority_is_1s": True}
    SETTINGS = {
        "distribution": {"default": "uniform", "sweep": ["uniform"]},
        "n_unique_elements": {
            "default": 2,
            "sweep": [2],
        },
    }
    XAXIS = "list_length"
    OTHER_AXES = ("n_unique_elements", "distribution")

    def generate_data(self, model_shorthand=None):
        domain = np.arange(2).reshape(1, -1).repeat(axis=0, repeats=self.settings["n_runs"])
        selected_points = self._rng.permuted(domain, axis=1)[:, : self.settings["n_unique_elements"]]
        modes = self._rng.permuted(selected_points, axis=1)[:, 0:1]
        n_repeats = (self.settings["list_length"]) / self.settings["n_unique_elements"]
        assert n_repeats.is_integer()
        concat_data = np.concatenate((selected_points.repeat(axis=1, repeats=n_repeats), modes), axis=1)
        data = self._rng.permuted(concat_data, axis=1)
        assert data.shape[1] == self.settings["list_length"] + 1
        assert data.shape[1] % 2 == 1
        instruction = "Output true if the majority of elements of this list are 1, else false:"
        inputs = [
            {
                "data": d.tolist(),
                "rendered_data": json.dumps(d.tolist()),
                "setting": self.settings,
                "instruction": instruction,
            }
            for d in data
        ]
        outputs = modes.tolist()

        return inputs, outputs


class IndexExperiment(MajorityExperiment):
    METRIC = "accuracy"
    OUTPUT_SCHEMA = {"element_value": -1}
    SETTINGS = {
        "task": {"default": "index_explicit", "sweep": ["index_explicit"]},
        "int_range": {"default": 200, "sweep": [200]},
    }
    XAXIS = "list_length"
    OTHER_AXES = ("int_range", "task")

    def generate_data(self, model_shorthand=None):
        n_runs = self.settings["n_runs"]
        universe = (
            self._rng.permuted(np.arange(self.settings["int_range"]), axis=0)
            .reshape(1, -1)
            .repeat(axis=0, repeats=n_runs)
        )
        left = universe[:, : self.settings["list_length"]].copy()
        indices = self._rng.integers(0, self.settings["list_length"], size=(n_runs, 1))
        outputs = np.take_along_axis(left, indices, axis=1).flatten().tolist()

        def render(d):
            if self.settings["task"] == "index":
                return json.dumps(d.tolist())
            elif self.settings["task"] == "index_explicit":
                return json.dumps([{"index": i, "value": v} for i, v in enumerate(d.tolist())])

        inputs = [
            {
                "data": {"left": l.tolist(), "id": idx.item()},
                "rendered_data": "List: " + render(l) + "\nIndex: " + str(idx.item()),
                "setting": self.settings,
                "instruction": "Output the element at the specified index (starting at 0) of the list:",
            }
            for l, idx in zip(left, indices)
        ]
        return inputs, outputs


class IntDisjointnessExperiment(MajorityExperiment):
    METRIC = "accuracy"
    OUTPUT_SCHEMA = {"is_disjoint": True}
    SETTINGS = {
        "task": {"default": "disjoint_dicts", "sweep": ["disjoint_lists_verbose"]},
        "int_range": {"default": 400, "sweep": [400]},
    }
    XAXIS = "list_length"
    OTHER_AXES = ("int_range", "task")

    def generate_data(self, model_shorthand=None):
        n_runs = self.settings["n_runs"]
        universe = self._2d_permutation(self.settings["int_range"], n_runs)
        left = universe[:, : self.settings["list_length"]].copy()
        right = universe[:, self.settings["list_length"] : 2 * self.settings["list_length"]].copy()
        # at this point, left and right are disjoint

        # add left element to right
        right[: n_runs // 2, 0] = left[: n_runs // 2, 0].copy()
        outputs = [set(l) & set(r) == set() for l, r in zip(left, right)]
        right = self._rng.permuted(right, axis=1)
        left = self._rng.permuted(left, axis=1)

        inputs = [
            {
                "data": {"left": l.tolist(), "right": r.tolist()},
                "rendered_data": "Left: " + json.dumps(l.tolist()) + "\nRight: " + json.dumps(r.tolist()),
                "setting": self.settings,
                "instruction": "Output true if left and right lists are disjoint (share no elements) and false otherwise:",
            }
            for l, r in zip(left, right)
        ]
        return inputs, outputs


class DisjointnessExperiment(IntDisjointnessExperiment):
    SETTINGS = IntDisjointnessExperiment.SETTINGS.copy()
    SETTINGS["task"] = {"default": "", "sweep": ["disjoint_lists_verbose"]}

    def generate_data(self, model_shorthand=None):
        n_runs = self.settings["n_runs"]
        # sample uniformly from (0, 0), (0, 1) and (1, 0)
        instances = self._rng.integers(0, 2, size=(n_runs, self.settings["list_length"]), endpoint=True)
        # set one entry to (1, 1)
        instances[n_runs // 2 :, self._rng.integers(0, self.settings["list_length"], size=n_runs // 2)] = 3
        left = instances % 2
        right = instances // 2
        outputs = [np.sum(l.astype(bool) & r.astype(bool)) == 0 for l, r in zip(left, right)]
        assert np.sum(outputs) == n_runs // 2

        def render(d):
            if self.settings["task"].startswith("disjoint_lists"):
                return json.dumps(d.tolist())
            elif self.settings["task"] == "disjoint_tuples":
                return json.dumps([(i, v) for i, v in enumerate(d.tolist())])
            elif self.settings["task"] == "disjoint_pseudo_json":
                return "[" + ", ".join([f"({i}, {v})" for i, v in enumerate(d.tolist())]) + "]"
            elif self.settings["task"] == "disjoint_dicts":
                return json.dumps(dict(enumerate(d.tolist())))
            elif self.settings["task"] == "disjoint_negative":
                return json.dumps([i + 1 if v == 1 else -i - 1 for i, v in enumerate(d.tolist())])

        if self.settings["task"] in ["disjoint_lists", "disjoint_dicts"]:
            instruction = "Output false if there is an i for which left[i]*right[i] == 1"
        elif self.settings["task"] == "disjoint_negative":
            instruction = "Output false if there is a number x > 0 that occurs in left and also in right."
        elif self.settings["task"] == "disjoint_lists_verbose":
            instruction = (
                "These left and right lists represent sets using binary indicators for each item. "
                "Output true if these sets are disjoint and false if they have a non-empty intersection. "
                "That is, output true if and only if there is no index where both lists contain 1."
            )
        else:
            instruction = "Output false if there is an i for which left[i][1]*right[i][1] == 1"
        inputs = [
            {
                "data": {"left": l.tolist(), "right": r.tolist()},
                "rendered_data": "\nLeft: " + render(l) + "\nRight: " + render(r),
                "setting": self.settings,
                "instruction": instruction,
            }
            for l, r in zip(left, right)
        ]
        return inputs, outputs


class Match2Experiment(MajorityExperiment):
    METRIC = "accuracy"
    OUTPUT_SCHEMA = {"found_i": True}
    SETTINGS = {
        "task": {"default": "match2", "sweep": ["match2"]},
        "int_range": {"default": 100, "sweep": [1000]},
    }
    XAXIS = "list_length"
    OTHER_AXES = ("int_range", "task")

    def _oracle(self, x, answer):
        x, y = x[:-1], -x[-1]
        return y in x

    def generate_data(self, model_shorthand=None):
        n_runs = self.settings["n_runs"]
        n_list = self.settings["list_length"]
        data = list()
        outputs = list()
        for answer in [True, False]:
            for _ in range(n_runs // 2):
                perm = self._rng.permutation(self.settings["int_range"]) - self.settings["int_range"] // 2
                x = perm[:n_list]
                y = -x[-1]

                if answer:
                    x[self._rng.integers(0, n_list - 1)] = y  # ensure y = C - x_n is in x
                else:
                    if y in x:  # ensure y = C - x_n is not in x
                        y_idx = np.flatnonzero(x == y)[0]
                        x[y_idx] = perm[n_list]
                data.append(x.astype(int).tolist())
                outputs.append(answer)
                assert self._oracle(x.astype(int).tolist(), answer) == answer

        return [
            {
                "data": d,
                "rendered_data": f"\nList: {json.dumps(d[:-1])}\nx: {d[-1]}",
                "setting": self.settings,
                "instruction": "You are given a list of numbers and a number x. "
                "Determine whether list[i] + x = 0 for some i.",
            }
            for d in data
        ], outputs


class Match3Experiment(MajorityExperiment):
    METRIC = "accuracy"
    OUTPUT_SCHEMA = {"found_i_and_j": True}
    SETTINGS = {
        "task": {"default": "match3", "sweep": ["match3"]},
        "int_range": {"default": 100, "sweep": [1000]},  # [200, 1000]},
    }
    XAXIS = "list_length"
    OTHER_AXES = ("int_range", "task")

    def _oracle(self, x):
        return self._has_pair(x, -x[-1])

    @classmethod
    def _has_pair(cls, l, target):
        numbers = set(l)
        for i in range(len(l)):
            if target - l[i] in numbers:
                return True
        return False

    def generate_data(self, model_shorthand=None):
        n_runs = self.settings["n_runs"]
        n_list = self.settings["list_length"]

        # Do rejection sampling to generate negative instances
        negative_instances = list()
        rejected_samples = 0

        while len(negative_instances) < n_runs:
            perm = self._rng.permutation(self.settings["int_range"] + 1) - self.settings["int_range"] // 4
            perm = perm[perm != 0]  # remove zero to avoid trivial solutions
            x = perm[:n_list]
            n = len(x) - 1
            if not self._has_pair(x, -x[n]):
                negative_instances.append(x.astype(int).tolist())
            else:
                rejected_samples += 1
            if rejected_samples > n_runs * 100:
                raise ValueError("Rejection rate > 99%. Stopping.")

        # Construct final instances
        data, outputs = list(), list()
        for answer in [True, False]:
            for _ in range(n_runs // 2):
                x = negative_instances.pop(0)
                outputs.append(answer)
                if not answer:
                    data.append({"val": x, "i": -1, "j": -1})
                else:
                    j = self._rng.integers(0, n_list - 1)
                    valid_idx = np.flatnonzero(x[:-1] != x[j])
                    i = self._rng.choice(valid_idx)
                    x[-1] = -(x[i] + x[j])
                    data.append({"val": x, "i": i, "j": j})

                assert self._oracle(x) == answer, f"Data: {data[-1]}, Answer: {answer}"

        return [
            {
                "data": d,
                "rendered_data": f"\nList: {json.dumps(d['val'][:-1])}\nx: {d['val'][-1]}",
                "setting": self.settings,
                "instruction": "You are given a list of numbers and a number x. "
                + "Determine whether list[i] + list[j] + x = 0 for some i, j.",
            }
            for d in data
        ], outputs


class ReachabilityExperiment(MajorityExperiment):
    METRIC = "accuracy"
    OUTPUT_SCHEMA = {"path_exists": True}
    SETTINGS = {
        "task": {"default": "reachability", "sweep": ["reachability"]},
        "n_paths": {"default": 2, "sweep": [2]},
    }
    XAXIS = "list_length"
    OTHER_AXES = ("n_paths", "task")

    @classmethod
    def _graph_to_text(cls, edges, nodes, s, t, use_json=True):
        output = f"You are given an directed graph with {len(nodes)} nodes as a list of edges (i, j).\n"
        output += "An edge (i,j) means that node i points to j.\n"
        output += "The edges in G are:\n"
        if use_json:
            output += json.dumps(edges) + "\n"
        else:
            output += ", ".join([f"({i}, {j})" for i, j in edges]) + "\n"
        output += f"Is there a path from node {s} to node {t}?\n"
        return output

    def _oracle(cls, data):
        # Extract edges using regex: matches pattern like "(number, number)"
        edges = re.findall(r"[\(\[]\s*(\d+)\s*,\s*(\d+)\s*[\)\]]", data)
        edges = [(int(u), int(v)) for u, v in edges]

        src_match = re.search(r"from node (\d+)", data)
        dst_match = re.search(r"to node (\d+)", data)
        src_node = int(src_match.group(1)) if src_match else None
        dst_node = int(dst_match.group(1)) if dst_match else None

        G = nx.DiGraph(edges)
        if src_node not in G.nodes() or dst_node not in G.nodes():
            return False
        return nx.has_path(G, src_node, dst_node)

    def generate_path_graph(self, connected):
        n_paths = self.settings["n_paths"]
        path_length = (self.settings["list_length"] // self.settings["n_paths"]) - 1

        paths = [(f"r{r}c{c}", f"r{r}c{c+1}") for c in range(path_length) for r in range(n_paths)]
        nodes = [f"r{r}c{c}" for c in range(path_length + 1) for r in range(n_paths)]
        mapping = {node: i for i, node in zip(self._rng.permutation(len(nodes)), nodes)}
        remapped_edges = self._rng.permutation([(mapping[i], mapping[j]) for i, j in paths]).astype(int).tolist()

        if not connected:
            r1, r2 = self._rng.choice(n_paths, size=2, replace=False)
        else:
            r1, r2 = [self._rng.choice(n_paths)] * 2
        s, t = mapping[f"r{r1}c0"], mapping[f"r{r2}c{path_length}"]
        return {
            "data": "",
            "rendered_data": self._graph_to_text(remapped_edges, nodes, s, t),
            "setting": self.settings,
            "instruction": f"",
        }, connected

    def generate_graphqa_graph(self, connected):
        sparsity = self._rng.uniform(0.0, 1.0)
        n_nodes = self.settings["list_length"]
        g = nx.erdos_renyi_graph(n_nodes, sparsity, seed=self._rng, directed=True)
        (
            s,
            t,
        ) = self._rng.choice(g.nodes(), size=2, replace=False)
        connected = nx.has_path(g, s, t)
        return {
            "data": "",
            "rendered_data": self._graph_to_text(list(g.edges()), list(g.nodes()), s, t),
            "setting": self.settings,
            "instruction": f"",
        }, connected

    def generate_data(self, model_shorthand=None, verify=True):
        inputs = []
        outputs = []
        for connected in [True, False]:
            for i in range(self.settings["n_runs"] // 2):
                if self.settings["task"] != "graphqa":
                    x, y = self.generate_path_graph(connected)
                else:
                    x, y = self.generate_graphqa_graph(connected)
                inputs.append(x)
                outputs.append(y)

        if verify:
            for input, output in zip(inputs, outputs):
                if self._oracle(input["rendered_data"]) != output:
                    raise ValueError("Oracle does not match the output")
        return inputs, outputs


class UniqueExperiment(MajorityExperiment):
    METRIC = "accuracy"
    OUTPUT_SCHEMA = {"unique": -1}
    SETTINGS = {
        "list_length": {"sweep": [6, 50, 100, 200, 500, 1000]},
        "task": {"default": "unique", "sweep": ["unique"]},
    }

    XAXIS = "list_length"  # "int_range"
    OTHER_AXES = ("int_range", "task")  # ("list_length", "task")

    @staticmethod
    def verify_unique_elements(data, outputs):
        for row, output in zip(data, outputs):
            freqs = np.bincount(row)
            if freqs[output] != 1:
                return False
            if (freqs == 1).sum() != 1:
                return False
        return True

    def generate_data(self, model_shorthand=None):
        list_len = self.settings["list_length"]
        n_qtr = list_len // 4
        random_ints = self._2d_permutation(list_len + 1, self.settings["n_runs"])
        # unique element is the first element after 1/4 * list_len
        unique_elements = random_ints[:, n_qtr : n_qtr + 1]
        # generate 1/2 * list_len elements that will occur >= 3 times
        three_or_more_times = self._rng.permuted(np.tile(random_ints[:, :n_qtr], (1, 10)), axis=1)[:, : n_qtr * 2]
        duplicates = np.tile(random_ints[:, :n_qtr], (1, 2))
        data = np.hstack((duplicates, three_or_more_times, unique_elements))
        data = self._rng.permuted(data, axis=1)
        assert self.verify_unique_elements(data, unique_elements)

        inputs = [
            {
                "data": d.tolist(),
                "rendered_data": json.dumps(d.tolist()),
                "setting": self.settings,
                "instruction": "Output the element in the list that occurs only once:",
            }
            for d in data
        ]
        return inputs, unique_elements.tolist()


class EqualityExperiment(MajorityExperiment):
    METRIC = "accuracy"
    OUTPUT_SCHEMA = {"equals": True}
    SETTINGS = {
        "task": {"default": "equality", "sweep": ["equality"]},
        "int_range": {"default": 2, "sweep": [2]},
    }
    XAXIS = "list_length"
    OTHER_AXES = ("int_range", "task")

    def generate_data(self, model_shorthand=None):
        n_runs = self.settings["n_runs"]
        left = self._rng.integers(0, self.settings["int_range"], size=(n_runs, self.settings["list_length"]))
        right = left.copy()
        outputs = [True] * n_runs

        for i in range(n_runs // 2):
            outputs[i] = False
            # swap two random elements, making sure they are not the same
            first_index = self._rng.integers(self.settings["list_length"])
            second_index = self._rng.choice(np.flatnonzero(left[i] != left[i, first_index]))
            right[i, first_index] = left[i, second_index]
            right[i, second_index] = left[i, first_index]
            assert (right[i] != left[i]).sum() == 2
        inputs = [
            {
                "data": {"left": l.tolist(), "right": r.tolist()},
                "rendered_data": "Left: " + json.dumps(l.tolist()) + "\nRight: " + json.dumps(r.tolist()),
                "setting": self.settings,
                "instruction": "Output true if the left and right lists are identical:",
            }
            for l, r in zip(left, right)
        ]
        return inputs, outputs


class SetDiffExperiment(MajorityExperiment):
    METRIC = "accuracy"
    OUTPUT_SCHEMA = {"element": -1}
    SETTINGS = {
        "separator": {"sweep": [True]},  # , False]},
        "list_length": {"sweep": [6, 50, 100, 200, 500, 1000]},
        "rel_b_size": {"sweep": [1]},
    }
    XAXIS = "list_length"
    OTHER_AXES = ("separator", "rel_b_size")

    @staticmethod
    def fair_split(budget, splits):
        in_running = list(range(len(splits)))
        sizes = [0] * len(splits)
        # first pass allocates minimums
        for i, split in enumerate(splits):
            allocated_size = (budget * split.get("weight", 0)) // sum([s.get("weight", 0) for s in splits])
            if allocated_size < split.get("min", 0):
                allocated_size = split["min"]
                in_running.remove(i)
                sizes[i] = allocated_size

        # second pass
        new_budget = budget - sum(sizes)
        total_weight = sum([splits[i].get("weight", 0) for i in in_running])
        for i in in_running:
            sizes[i] = (new_budget * splits[i].get("weight", 0)) // total_weight

        # if there is any remaining budget, allocate it to the first split
        i = 0
        while budget - sum(sizes) > 0:
            sizes[in_running[i]] += 1
            i = (i + 1) % len(in_running)
        return sizes

    def generate_data(self, model_shorthand=None):
        list_len = self.settings["list_length"]
        random_ints = self._2d_permutation(list_len, self.settings["n_runs"])
        # unique element is the first element after 1/4 * list_len
        sizes = self.fair_split(
            list_len, [{"weight": 2, "min": 2}, {"min": 1}, {"weight": self.settings["rel_b_size"]}, {"min": 1}]
        )
        sizes[0] = sizes[0] // 2

        shared_set = random_ints[:, : sizes[0]]
        unique_elements = random_ints[:, sizes[0] : sizes[0] + sizes[1]]
        only_in_b = random_ints[:, sizes[0] + sizes[1] : sizes[0] + sizes[1] + sizes[2]]

        outputs = unique_elements.flatten()
        outputs[: len(outputs) // 2] = -1
        outputs = outputs.tolist()

        fill_b = unique_elements.copy()
        fill_b[len(fill_b) // 2 :] = random_ints[len(fill_b) // 2 :, 0:1]

        set_a = np.hstack((shared_set, unique_elements))
        set_b = np.hstack((shared_set, only_in_b, fill_b))
        set_a = self._rng.permuted(set_a, axis=1)
        set_b = self._rng.permuted(set_b, axis=1)

        data = [{"A": a.tolist(), "B": b.tolist()} for a, b in zip(set_a, set_b)]
        for d, o in zip(data, outputs):
            if o == -1:
                assert set(d["A"]) - set(d["B"]) == set(), f"Data: {d}, Answer: {o}"
            else:
                assert set(d["A"]) - set(d["B"]) == {o}, f"Data: {d}, Answer: {o}"

        inputs = list()
        for d in data:
            if self.settings["separator"]:
                rendered = f"Set A: {d['A']}\nSet B: {d['B']}"
                instructions = "You are given two sets of numbers A and B. "
            else:
                rendered = f"List: {json.dumps(d['A'] + d['B'])}"
                instructions = "You are given a list of numbers that contains two sets A and B."
                if self.settings["rel_size_of_B"] == 1:
                    instructions += " The first half of the list is set A and the second half of the list is set B. "
                else:
                    instructions += f" The first {len(d['A'])} elements are set A and the rest of the list is set B. "
            instructions += "Output the element in set A that is not in set B. If there is no such element, output -1."
            inputs += [{"data": d, "rendered_data": rendered, "setting": self.settings, "instruction": instructions}]
        return inputs, outputs
