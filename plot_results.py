import pathlib
import re
import plotly.express as px
import polars as pl
import numpy as np
import scipy.stats
import colour
import click
from pandas.io.clipboard import copy


XAXIS = "list_length"
METRIC = "accuracy"
PROBLEMS = {
    "synthetic_main": [
        "Index",
        "Equality",
        "Match2",
        "Reachability",
        "Majority",
        "Match3",
    ],
    "synthetic_bonus": [
        "Disjointness",
        "IntDisjointness",
    ],
    "synthetic_bonus_sigma_hard": [
        "Unique",
        "SetDiff",
    ],
    "real_world": ["MostNegativeReview", "MajorityReview", "VariableTracking"],
}


def _conf_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return h


def translate(x, suffix):
    name = x.text.split("=")[-1]
    lut = {
        "MostNegativeReview": "FindNegativeReview <br>(Index / Needle-In-Haystack)",
        "MajorityReview": "Majority Review<br>(Majority)",
        "VariableTracking": "Variable Tracking<br>(Reachability)",
    }
    return lut.get(name, name) + suffix


def plot_experiments(
    df_full,
    filename,
    which,
    show_plot=False,
    save_plot=True,
):
    # add jitter to the x-axis for each model to improve readability
    jitter = df_full["pretty_name"].unique(maintain_order=True).to_frame("pretty_name").with_row_index("offset")
    df_full = (
        df_full.join(jitter, on="pretty_name", how="left")
        .with_columns(
            (pl.col(XAXIS) + pl.col("offset") * 2 - 4).alias(XAXIS),
            pl.col("class_name").str.replace("Experiment", ""),
        )
        .filter(pl.col("class_name").is_in(which))
    )

    df = df_full.group_by("pretty_name", XAXIS, "class_name", maintain_order=True).agg(
        [
            pl.col("metric_value").mean().alias(METRIC),
            pl.col("metric_value").map_batches(_conf_interval).alias("conf_interval"),
        ]
    )
    suffix = df_full["suffix"].item(0) if "suffix" in df_full.columns else ""
    order = [
        "GPT-4o",
        "GPT-4o mini",
        "Claude 3.5 Sonnet",
        "Claude 3.5 Haiku",
        "Gemini 1.5 Pro",
        "Gemini 1.5 Flash",
        "o3",
        "Gemini 2.5 Flash",
    ]
    my_palette = list()
    for color in px.colors.qualitative.D3:
        my_palette.append(color)
        # use regex to extract from rgb('r, g, b) to hex
        if "rgb" in color:
            rgb = tuple(map(lambda x: int(x) / 255.0, re.findall(r"\d+", color)))
            orig = colour.Color(rgb=rgb)
        else:
            orig = colour.Color(color)
        orig.set_luminance(orig.get_luminance() + 0.18)
        orig.set_saturation(max(orig.get_saturation() - 0.1, 0))
        my_palette.append(orig.hex)
    fctr = 4 if len(which) == 4 else 3
    fig = px.line(
        df,
        x=XAXIS,
        y=METRIC,
        error_y="conf_interval",
        color="pretty_name",
        color_discrete_sequence=my_palette,
        markers="pretty_name",
        line_dash="pretty_name",
        line_dash_sequence=["solid", "dot"] * 10,
        facet_col="class_name",
        facet_col_wrap=min(fctr, len(which)),
        subtitle="",
        range_y=[0.001, 1.05],
        category_orders={"class_name": which, "pretty_name": order},
        labels={"pretty_name": "model", "list_length": "<i>n</i>"},
        template="simple_white",
        width=1200,
        height=300 * max(1, len(which) // fctr) - 30,
        facet_row_spacing=0.15,
    )
    fig.update_xaxes(title_standoff=0, showticklabels=True)
    fig.update_traces(line=dict(width=2.0), error_y=dict(thickness=0.5), marker=dict(size=4.5))
    extra = dict(y=1) if len(which) != 3 else dict()
    fig.update_layout(
        legend=dict(yanchor="top", **extra),
        margin=dict(l=20, r=20, t=50 if "real" in str(filename) else 30, b=0),
    )

    fig.for_each_annotation(lambda a: a.update(text=translate(a, suffix), font={"variant": "small-caps"}))
    _render(fig, filename, save_plot, show_plot)


def _render(fig, filename, save_plot, show_plot):
    fig.update_layout(
        font_size=16,
        font_color="black",
        font_family="NimbusRomNo9L-Reg",
        title={"yref": "paper", "y": 1, "pad": {"b": 15}, "yanchor": "bottom", "font": {"variant": "small-caps"}},
    )
    if save_plot:
        print(f"Saving {filename} ...")
        pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(filename)
    if show_plot:
        fig.show()


def plot_token_costs(input_dir, output_dir, which="synthetic_main", show_plot=False, save_plot=True):
    filename = f"{output_dir}/token_costs.pdf"
    df_full = batch_read(input_dir, "reasoning")
    normalized = df_full.with_columns(
        pl.col("class_name").str.replace("Experiment", ""), pl.col("costs").struct.field("reasoning")
    ).filter(pl.col("class_name").is_in(PROBLEMS[which]))
    fig = px.box(
        normalized,
        x="list_length",
        y="reasoning",
        color="pretty_name",
        facet_col="class_name",
        category_orders={"class_name": PROBLEMS[which]},
        facet_col_wrap=3,
        color_discrete_sequence=px.colors.qualitative.D3,
        template="simple_white",
        labels={"pretty_name": "model", "reasoning": "reasoning tokens", "list_length": "<i>n</i>"},
        width=1200,
        height=300 * 2,
        facet_row_spacing=0.1,
        log_y=True,
        range_y=[10, 100000],
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
    )
    fig.update_yaxes(dtick=1)
    fig.update_xaxes(tickvals=[10, 50, 100, 200])
    fig.for_each_annotation(lambda a: a.update(text=translate(a, ""), font={"variant": "small-caps"}))
    _render(fig, filename, save_plot, show_plot)


def draw_examples(input_dir, ordering):
    df_full = batch_read(input_dir, "no_CoT")
    examples = (
        df_full.filter(pl.col("pretty_name") == "GPT-4o")
        .with_columns(pl.col("class_name").str.replace("Experiment", ""))
        .sort(XAXIS)
        .group_by("class_name", maintain_order=True)
        .agg(
            pl.col("llm_request").first().str.strip_chars().str.head(500),
            pl.col("llm_response").first().str.strip_chars(),
        )
    )
    # reorder via join
    copy(
        examples.join(pl.DataFrame({"class_name": ordering}), on="class_name", how="right", maintain_order="right")
        .select("class_name", "llm_request", "llm_response")
        .to_pandas()
        .to_latex(index=False, escape=True)
    )


def plot_problems(input_dir, output_dir, what, show_plot=False):
    for cot in ["no_CoT", "with_CoT"]:
        if what == "real_world" and cot == "with_CoT":
            continue
        df_full = batch_read(input_dir, cot)
        if what == "synthetic_main" and cot == "with_CoT":
            df_reason = batch_read(input_dir, "reasoning").filter(pl.col("y_pred") != "[]")
            df_full = pl.concat([df_full, df_reason], how="diagonal")
        plot_experiments(df_full, f"{output_dir}/{what}_{cot}.pdf", which=PROBLEMS[what], show_plot=show_plot)


def batch_read(input_dir, cot):
    dfs = []
    for path in pathlib.Path().glob(f"{input_dir}/{cot}/results*.jsonl"):
        print(f"Loading {path} ...")
        dfs.append(pl.read_ndjson(path))
    df_full = pl.concat(dfs, how="diagonal")
    return df_full


@click.command()
@click.option('--input-dir', prompt='Input directory', default='results', show_default=True)
@click.option('--output-dir', prompt='Output directory', default='C://results', show_default=True)
def main(input_dir, output_dir):
    plot_token_costs(input_dir, output_dir)

    for what in PROBLEMS.keys():
        plot_problems(input_dir, output_dir, what, show_plot=False)

    draw_examples(input_dir, sum(list(PROBLEMS.values())[:3], []))


if __name__ == "__main__":
    main()
