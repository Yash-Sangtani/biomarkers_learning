import json
import os
import pathlib

import numpy as np
import pandas as pd
from tqdm import tqdm

pd.options.display.float_format = "{:}".format

DATAFRAME_HEAD_COUNT = 3


def convert_and_save_timestamps(input_path, column_names, timestamp_offset):
    output_path = input_path.with_name(input_path.stem + "_unix_datetime").with_suffix(
        input_path.suffix
    )

    df = pd.read_csv(input_path)

    for column_name in column_names:
        unix_column_name = column_name + "_unix"
        datetime_column_name = column_name + "_datetime"

        df[unix_column_name] = df[column_name] + timestamp_offset
        df[datetime_column_name] = pd.to_datetime(df[unix_column_name], unit="s")

    df.to_csv(output_path)


for dir in tqdm(os.listdir("./data/")):
    if not os.path.isdir(f"./data/{dir}/eye"):
        continue

    rec_dir = pathlib.Path(f"./data/{dir}/eye/")
    export_dir = rec_dir.joinpath("exports").joinpath("001")

    with rec_dir.joinpath("info.player.json").open() as file:
        meta_info = json.load(file)

    start_timestamp_unix = meta_info["start_time_system_s"]
    start_timestamp_pupil = meta_info["start_time_synced_s"]
    start_timestamp_diff = start_timestamp_unix - start_timestamp_pupil

    convert_and_save_timestamps(
        export_dir.joinpath("fixations.csv"),
        ["start_timestamp"],
        timestamp_offset=start_timestamp_diff,
    )
    convert_and_save_timestamps(
        export_dir.joinpath("blinks.csv"),
        ["start_timestamp"],
        timestamp_offset=start_timestamp_diff,
    )
    convert_and_save_timestamps(
        export_dir.joinpath("pupil_positions.csv"),
        ["pupil_timestamp"],
        timestamp_offset=start_timestamp_diff,
    )
