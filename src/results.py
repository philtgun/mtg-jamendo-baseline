import csv
from pathlib import Path

import pandas as pd


# Intentionally not using pandas here, so it will work with whatever values are
def add_to_output(path: Path, values: list) -> None:
    with path.open('a') as fp:
        writer = csv.writer(fp)
        writer.writerow(values)


def parse_output(path: Path) -> None:
    df = pd.read_csv(path, header=None)
    print(df.groupby(1).mean())
    print(df.groupby(1).std())
