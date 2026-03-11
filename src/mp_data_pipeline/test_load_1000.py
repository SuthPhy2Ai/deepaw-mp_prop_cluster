"""Create a 1000-record ASE test database from the JSONL.gz dataset."""

from .config import RAW_JSONL_PATH, TEST_DB_PATH
from .load_from_jsonl import load_jsonl_to_ase
from .utils import setup_logging


def main() -> None:
    setup_logging()
    load_jsonl_to_ase(
        jsonl_path=RAW_JSONL_PATH,
        db_path=TEST_DB_PATH,
        overwrite=True,
        limit=1000,
        log_every=100,
    )


if __name__ == "__main__":
    main()
