"""Load data/telco.csv into the telco.customers table.

Usage:
    python -m ingestion.load_telco
    python -m ingestion.load_telco --file data/telco.csv
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DEFAULT_CSV_PATH = Path("data/Telco_Customer_Churn.csv")
TABLE_NAME = "customers"
SCHEMA_NAME = "telco"


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set.")
    return database_url


def to_snake_case(column: str) -> str:
    column = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", column)
    return column.strip().lower().replace(" ", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake_case(col) for col in df.columns]
    return df


def load_customers(csv_path: Path, database_url: str) -> tuple[int, int]:
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    engine = create_engine(database_url)
    try:
        # replace makes re-running the load idempotent.
        df.to_sql(TABLE_NAME, engine, schema=SCHEMA_NAME, if_exists="replace", index=False)
    finally:
        engine.dispose()

    return len(df), len(df.columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load telco.csv into telco.customers.")
    parser.add_argument("--file", type=Path, default=DEFAULT_CSV_PATH, help="Path to telco.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.file.exists():
        raise FileNotFoundError(f"{args.file} not found")

    row_count, column_count = load_customers(args.file, get_database_url())
    print(f"{args.file.name}: {row_count} rows loaded into {SCHEMA_NAME}.{TABLE_NAME} ({column_count} columns)")


if __name__ == "__main__":
    main()
