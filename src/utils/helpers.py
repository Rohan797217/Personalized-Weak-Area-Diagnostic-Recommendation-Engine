"""
Utility helpers for the AptiDude DS project.

Provides:
- Path management (project root, data directories)
- Logging setup
- Neon PostgreSQL database connection via SQLAlchemy
- Common data loading functions
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root is two levels up from src/utils/helpers.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "DATASET"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

# Ensure output directories exist
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env at project root
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Database connection (Neon PostgreSQL)
# ---------------------------------------------------------------------------


def get_database_url() -> str:
    """
    Return the DATABASE_URL from environment.
    Falls back to a placeholder if not set.
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "DATABASE_URL not found in environment. "
            "Create a .env file with: DATABASE_URL=postgresql://user:pass@host/db?sslmode=require"
        )
    return url


def get_sqlalchemy_engine():
    """Create a SQLAlchemy engine connected to Neon PostgreSQL."""
    from sqlalchemy import create_engine

    url = get_database_url()
    engine = create_engine(url, echo=False, pool_pre_ping=True)
    return engine


def upload_df_to_db(df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
    """Upload a DataFrame to Neon PostgreSQL."""
    engine = get_sqlalchemy_engine()
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    logger = get_logger("helpers")
    logger.info(f"Uploaded {len(df)} rows to table '{table_name}'")


def read_table_from_db(table_name: str) -> pd.DataFrame:
    """Read an entire table from Neon PostgreSQL."""
    engine = get_sqlalchemy_engine()
    return pd.read_sql_table(table_name, engine)


def run_query(query: str) -> pd.DataFrame:
    """Run a SQL query against Neon PostgreSQL and return a DataFrame."""
    engine = get_sqlalchemy_engine()
    return pd.read_sql(query, engine)


# ---------------------------------------------------------------------------
# Data loading (from local CSV files)
# ---------------------------------------------------------------------------


def load_raw_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from the DATASET directory."""
    path = DATASET_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path)


def load_processed_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from the data/processed directory."""
    path = DATA_PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_csv(path)


def save_processed_csv(df: pd.DataFrame, filename: str):
    """Save a DataFrame to the data/processed directory."""
    path = DATA_PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    logger = get_logger("helpers")
    logger.info(f"Saved {len(df)} rows to {path}")
