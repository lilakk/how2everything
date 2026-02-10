#!/usr/bin/env python3
"""Load an Arrow file and print the first record."""

import argparse
import json
import pyarrow as pa


def main() -> None:
    parser = argparse.ArgumentParser(description="Load an Arrow file and print the first record.")
    parser.add_argument("path", type=str, help="Path to an Arrow (.arrow) file.")
    args = parser.parse_args()

    with pa.memory_map(args.path, "r") as source:
        reader = pa.ipc.open_file(source)
        table = reader.read_all()

    print(f"Schema: {table.schema}")
    print(f"Rows: {table.num_rows}, Columns: {table.num_columns}")
    print()

    first = {col: table.column(col)[0].as_py() for col in table.column_names}
    print(json.dumps(first, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
