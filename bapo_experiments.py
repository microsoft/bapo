import collections
import pathlib

from sammo.data import DataTable

import polars as pl
import pandas as pd

import utils
from exps.synthetic import *
from exps.realworld import *


def run_experiment(
    experiment=Match2Experiment,
    mode="default",
    all_models=["4o-mini", "4o", "g-1.5-pro", "g-1.5-flash", "sonnet-3.5", "haiku-3.5"],
    n_runs=10,
    save_file=True,
):
    suffix = {"cot": "with_CoT", "reasoning": "reasoning", "default": "no_CoT"}[mode]
    filename = pathlib.Path(f"results/{suffix}/results_{experiment.name().lower()}_{suffix}_{n_runs}_runs.jsonl")
    if filename.exists():
        df = pl.read_ndjson(filename)
        if set(all_models).issubset(df["model"].unique()):
            print(f"Skipping. Found results file for {experiment.name()} with {n_runs} runs.")
            return df

    shared_settings = collections.defaultdict(dict)
    shared_settings["model"]["default"] = "4o-mini"
    shared_settings["model"]["sweep"] = all_models

    shared_settings["n_runs"]["default"] = n_runs
    shared_settings["seed"]["default"] = 42
    shared_settings["list_length"] = {
        "default": 10,
        "sweep": [6, 50, 100, 200] if experiment != EqualityExperiment else [10, 50, 100, 200],
    }

    shared_settings["cot"]["default"] = mode == "cot"
    shared_settings["suffix"]["default"] = " (w/ CoT)" if mode == "cot" else ""

    runners = collections.defaultdict(list)
    for setting in experiment.sweep(**shared_settings):
        runners[setting["model"]].append(setting)

    all_results = list()
    for model_shorthand, settings in runners.items():
        print(f"Running experiment for {model_shorthand}")

        extra_kwargs = {}
        if experiment == SetDiffExperiment and model_shorthand.startswith("g-"):
            extra_kwargs = {"retry": False}
        runner = utils.get_runner(model_shorthand, **extra_kwargs)
        pretty_name = utils.lookup_model(model_shorthand)["pretty_name"]

        model_inputs = list()
        model_outputs = list()
        for setting in settings:
            inputs, outputs = experiment(**setting).generate_data(model_shorthand)

            # verify our data generation is deterministic
            for i in range(10):
                _inputs, _outputs = experiment(**setting).generate_data(model_shorthand)
                assert _outputs == outputs
                assert all(x["rendered_data"] == y["rendered_data"] for x, y in zip(_inputs, inputs))

            model_inputs += inputs
            model_outputs += outputs

        df_true = DataTable(model_inputs, model_outputs)
        df_pred = experiment.transform(runner, df_true)

        current_run = experiment.metric(df_true, df_pred, pretty_name=pretty_name, class_name=experiment.__name__)
        all_results += current_run

    df = pl.DataFrame(pd.DataFrame(all_results))
    if save_file:
        filename.parent.mkdir(parents=True, exist_ok=True)
        df.write_ndjson(filename)
    return df


def real_world_experiments(
    experiments=[VariableTrackingExperiment, MajorityReviewExperiment, MostNegativeReviewExperiment]
):
    for exp in experiments:
        print(f"\nRunning {exp.__name__}")
        run_experiment(exp, n_runs=100, mode="default")


def synthetic_experiments(
    experiments=[
        ReachabilityExperiment,
        IndexExperiment,
        EqualityExperiment,
        MajorityExperiment,
        Match2Experiment,
        Match3Experiment,
        MinExperiment,
        MaxExperiment,
        IntDisjointnessExperiment,
        DisjointnessExperiment,
        UniqueExperiment,
        SetDiffExperiment,
    ]
):
    for cot in (False, True):
        for exp in experiments:
            print(f"\nRunning {exp.__name__} with CoT={cot}")
            run_experiment(exp, n_runs=100, mode="cot" if cot else "default")


def reasoning_experiments(
    experiments=[
        IndexExperiment,
        EqualityExperiment,
        Match2Experiment,
        ReachabilityExperiment,
        MajorityExperiment,
        Match3Experiment,
    ],
    models=["g-2.5-flash", "o3"],
):
    for exp in experiments:
        print(f"\nRunning {exp.__name__}")
        run_experiment(exp, n_runs=100, mode="reasoning", all_models=models)


def teaser_experiments(
    experiments=[ReachabilityExperiment, IndexExperiment], models=["GPT-4o", "Gemini 1.5 Pro", "Claude 3.5 Sonnet"]
):
    for exp in experiments:
        print(f"\nRunning {exp.__name__}")
        df_full = run_experiment(exp, n_runs=100, mode="default")
        exp.plot_single_experiment(df_full, models=models)


if __name__ == "__main__":
    _ = sammo.setup_logger(log_prompts_to_file=True)
    reasoning_experiments()
    teaser_experiments()
    real_world_experiments()
    synthetic_experiments()
