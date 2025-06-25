import itertools
import json
import random
import re
import warnings

import numpy as np
import polars as pl
import sammo.utils
import scipy.stats
from plotly import express as px
from sammo.base import Template, Costs
from sammo.components import GenerateText, Output
from sammo.extractors import JSONPath


class Experiment:
    OUTPUT_SCHEMA = {"computed_value": -1}
    NORMALIZE_OUTPUT = True

    def __init__(self, **my_settings):
        merged_settings = {
            **{k: v["default"] if "default" in v else v["sweep"][0] for k, v in self.SETTINGS.items()},
            **my_settings,
        }
        list_len = merged_settings["list_length"]
        seed = list_len if isinstance(list_len, int) else sum(list_len)
        self._rng = np.random.default_rng(merged_settings["seed"] + seed)
        self._pyrng = random.Random(merged_settings["seed"] + seed)
        self.settings = merged_settings

    @classmethod
    def sweep(cls, **shared_settings):
        all_settings = shared_settings | cls.SETTINGS
        sweepable = {k: v["sweep"] for k, v in all_settings.items() if "sweep" in v}
        constant = {k: v["default"] for k, v in all_settings.items() if "sweep" not in v}

        for i, values in enumerate(itertools.product(*sweepable.values())):
            yield {**dict(zip(sweepable.keys(), values)), **constant}

    @classmethod
    def name(self):
        return self.__name__.replace("Experiment", "")

    def generate_data(self, model_shorthand=None):
        raise NotImplementedError

    def _2d_permutation(self, high, n_rows):
        return self._rng.permuted(np.tile(np.arange(int(high)), (n_rows, 1)), axis=1)

    @classmethod
    def _clean_filename(cls, suffix):
        return re.sub(r"[^a-zA-Z0-9()]", "_", suffix)

    @staticmethod
    def _conf_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return h

    @classmethod
    def plot_single_experiment(cls, df_full, show_plot=True, models=None):
        # add offsets to the x-axis for each model to improve readability
        offsets = df_full["pretty_name"].unique(maintain_order=True).to_frame("pretty_name").with_row_index("offset")
        df_full = df_full.join(offsets, on="pretty_name", how="left").with_columns(
            (pl.col(cls.XAXIS) + pl.col("offset") * 2 - 4).alias(cls.XAXIS)
        )

        df = df_full.group_by("pretty_name", cls.XAXIS, *cls.OTHER_AXES, maintain_order=True).agg(
            [
                pl.col("metric_value").mean().alias(cls.METRIC),
                pl.col("metric_value").map_batches(cls._conf_interval).alias("conf_interval"),
            ]
        )
        if models:
            df = df.filter(pl.col("pretty_name").is_in(models))

        suffix = df_full["suffix"].item(0) if "suffix" in df_full.columns else ""
        if df[cls.OTHER_AXES[0]].n_unique() > 1 or df[cls.OTHER_AXES[1]].n_unique() > 1:
            raise ValueError("Paper plot only accepts one axis with varying values.")
        fig = px.line(
            df,
            x=cls.XAXIS,
            y=cls.METRIC,
            error_y="conf_interval",
            color="pretty_name",
            markers="pretty_name",
            line_dash="pretty_name",
            title=None,  # cls.__name__.replace("Experiment", "") + suffix,
            range_y=[0.001, 1.05],
            labels={"pretty_name": "model", "list_length": "$n$"},
            template="simple_white",
            width=500,
            height=300,
        )
        fig.update_xaxes(title_standoff=0)
        fig.update_layout(
            legend=dict(orientation="h", y=0.4, x=0.05, title_text=""),
            font_size=28,
            font_color="black",
            font_family="NimbusRomNo9L-Reg",
            title={"yref": "paper", "y": 1, "pad": {"b": 15}, "yanchor": "bottom", "font": {"variant": "small-caps"}},
            margin=dict(l=20, r=20, t=30, b=20),
        )

        filename = cls._clean_filename(cls.__name__.replace("Experiment", "").lower() + suffix) + ".pdf"
        path = sammo.utils.MAIN_PATH / "results" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(path)

        if show_plot:
            fig.show()

    @classmethod
    def plot_set(cls, df_full):
        df = df_full.group_by("pretty_name", cls.XAXIS, *cls.OTHER_AXES, maintain_order=True).agg(
            pl.col("metric_value").mean()
        )
        combinations = df[cls.OTHER_AXES[3:]].unique().to_dicts()
        if len(combinations) == 0:
            combinations = [{}]

        for value in combinations:
            fig = px.line(
                (df.filter(**value) if len(value) > 0 else df),
                x=cls.XAXIS,
                y="metric_value",
                color=cls.OTHER_AXES[0],
                facet_col=cls.OTHER_AXES[1] if len(cls.OTHER_AXES) > 1 else None,
                facet_row=cls.OTHER_AXES[2] if len(cls.OTHER_AXES) > 2 else None,
                template="simple_white",
                markers=cls.OTHER_AXES[0],
                title=cls.__name__.replace("Experiment", "")
                + "("
                + ", ".join(f"{k}={v}" for k, v in value.items())
                + ")",
            )
            fig.update_layout(font_size=14)
            fig.show()

    @classmethod
    def transform(cls, runner, df):
        if isinstance(runner, str):
            return cls.baseline(runner, df)

        scratch = {s["cot"] for s in df.inputs.field("setting").inputs.values}
        if len(scratch) > 1:
            raise ValueError(f"Multiple cot settings found: {scratch}")
        use_scratchpad = scratch.pop() if scratch else False

        if use_scratchpad:
            schema = {"cot": ""} | cls.OUTPUT_SCHEMA
        else:
            schema = cls.OUTPUT_SCHEMA
        preamble = "" if not use_scratchpad else f"Think step by step on the cot, but stay under 250 words.\n"
        answer = GenerateText(
            Template(preamble + "{{input.instruction}} {{input.rendered_data}}"),
            json_mode=runner.guess_json_schema(schema),
        )
        extracted_value = JSONPath(answer, path=f"$..{list(cls.OUTPUT_SCHEMA.keys())[-1]}")
        return Output(extracted_value).run(runner, df)

    @classmethod
    def _accuracy(cls, y_true, y_pred, na_value=-1000):
        results = []
        for y_p, y_t in zip(y_pred, y_true):
            y_t = int(y_t)
            try:
                y_p = int(y_p)
            except (ValueError, TypeError):
                warnings.warn(f"Invalid prediction value: {y_p}. Setting to {na_value}.")
                y_p = na_value
            results.append(y_t == y_p)
        return results

    @classmethod
    def metric(cls, df_true, df_pred, **extra):
        y_pred = df_pred.outputs.normalized_values()
        y_true = df_true.outputs.normalized_values()

        errors = cls._accuracy(y_true, y_pred)
        costs = [sum([r.costs for r in row], Costs()).to_dict() for row in df_pred.outputs.llm_results]
        return [
            {
                "metric_value": errors[i],
                "metric_name": cls.METRIC,
                "y_true": str(y_true[i]),
                "y_pred": str(y_pred[i]),
                "costs": costs[i],
                "llm_request": str(df_pred.outputs.llm_requests[i][0] if df_pred.outputs.llm_requests[i] else ""),
                "llm_response": json.dumps(
                    df_pred.outputs.llm_responses[i][0] if df_pred.outputs.llm_responses[i] else ""
                ),
                **df_true.inputs.values[i]["setting"],
                **extra,
            }
            for i in range(len(y_true))
        ]
