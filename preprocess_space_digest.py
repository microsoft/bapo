import json
import tarfile
import zipfile
import polars as pl
from pathlib import Path
import polars.selectors as cs

from sammo.base import Template
from sammo.components import GenerateText, Output

import utils

CWD = Path(__file__).parent
RAW_DATA_DIR = CWD.parent / "raw_data"
ZIP_FILE = RAW_DATA_DIR / "space_digest.zip"
TAR_FILE = RAW_DATA_DIR / "space.tar.gz"
AGGREGATE_FILE = RAW_DATA_DIR / "space_aggregate.json"

OUTPUT_FILE = CWD.parent / "processed_data" / "space.json"


def load_json_from_tar(fp, json_file_name):
    with tarfile.open(fp, "r:gz") as tar:
        return pl.read_json(tar.extractfile(json_file_name))


def load_jsonl_from_zip(fp, json_file_name):
    with zipfile.ZipFile(fp, "r") as z:
        with z.open(json_file_name) as f:
            return pl.read_ndjson(f.read())


def preprocess(force_fresh=False):
    if not force_fresh and AGGREGATE_FILE.exists():
        print("Aggregate file already exists, skipping preprocessing.")
        return
    print("Loading space digest...")
    space_digest = pl.concat(
        [
            load_jsonl_from_zip(ZIP_FILE, "space_digest/validation.jsonl"),
            load_jsonl_from_zip(ZIP_FILE, "space_digest/test.jsonl"),
        ],
    )
    ids = space_digest["id"].unique(maintain_order=True)
    print(f"Found {len(ids)} unique ids.")

    print("Loading space raw...")
    df = load_json_from_tar(TAR_FILE, "space_train.json")
    space = (
        df.filter(pl.col("entity_id").is_in(ids))
        .explode("reviews")
        .with_columns(pl.col("reviews").struct.unnest())
        .drop("reviews")
        .with_columns(pl.col("rating").cut([1, 4], labels=["-1", "0", "1"]).alias("label").cast(int))
        .with_columns(pl.col("sentences").list.join(" ").alias("review"))
        .filter(pl.col("review").str.len_chars() <= 1000)
        .pivot("label", index="entity_id", values="review", aggregate_function=pl.element())
    )
    print(space["entity_id"].n_unique(), " unique ids in space.")
    space.write_json(AGGREGATE_FILE)


def quality_check(min_positive=105, min_negative=55, slack=1.5):
    OUTPUT_FORMAT = {"answers": [{"id": 42, "label": {"-1", "0", "1"}}]}

    # annotate slack * min reviews to account for loss due to inconsistent labels
    space = (
        pl.read_json(AGGREGATE_FILE)
        .filter(pl.col("1").list.len() >= min_positive, pl.col("-1").list.len() >= min_negative)
        .with_columns(
            cs.by_name("-1", "0").list.head(int(min_negative * slack)), pl.col("1").list.head(int(min_positive * slack))
        )
    )
    print(
        f"Filtered to {len(space)} entities with at least {min_positive} "
        f"positive reviews and {min_negative} negative reviews."
    )

    records = space.unpivot(
        ["-1", "0", "1"], index="entity_id", variable_name="orig_label", value_name="reviews"
    ).to_dicts()

    runner = utils.get_runner("4.1-nano")
    json_schema = runner.guess_json_schema(OUTPUT_FORMAT)
    formatted_inputs = [
        json.dumps([{"id": i, "review": r} for i, r in enumerate(record["reviews"] or [])], ensure_ascii=False)
        for record in records
    ]
    dtable = Output(
        GenerateText(
            Template("For each review, decide whether it is positive (1), neutral (0), or negative (-1): {{input}}"),
            json_mode=json_schema,
        )
    ).run(runner, formatted_inputs)
    output_records = [v["answers"] for v in dtable.outputs.values]

    all_reviews = list()
    for i in range(len(records)):
        if output_records[i]:
            consistent_reviews = (
                pl.concat((pl.DataFrame(records[i]).with_row_index("id"), pl.DataFrame(output_records[i])), how="align")
                .filter(pl.col("orig_label") == pl.col("label"))
                .drop("id")
            )
            print("Dropped", len(records[i]["reviews"] or []) - len(consistent_reviews), "inconsistent reviews.")
            all_reviews.append(consistent_reviews)
    all_reviews = pl.concat(all_reviews)
    pivoted = all_reviews.pivot("orig_label", index="entity_id", values="reviews", aggregate_function=pl.element())

    print(pivoted.with_columns(pl.col("1").list.len(), pl.col("-1").list.len()))
    pivoted = pivoted.filter(pl.col("1").list.len() >= min_positive, pl.col("-1").list.len() >= min_negative).sort(
        "entity_id"
    )
    print("Final number of entities:", len(pivoted))
    pivoted.write_json(OUTPUT_FILE)


if __name__ == "__main__":
    preprocess(force_fresh=False)
    quality_check(min_negative=53, min_positive=101, slack=1.3)
