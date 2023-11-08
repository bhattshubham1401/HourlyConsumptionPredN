from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path
    local_data_file1: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_dir: Path
    STATUS_FILE: str
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_dir: Path

