"""Чтение исходного CSV и JSONL без pandas и без изменения исходных данных."""

import csv
import json
from pathlib import Path
from .models import Profile

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "hackathon-dataset-anonymized.jsonl"


class DatasetError(ValueError):
    """Поврежденный файл или запись с нарушенным контрактом."""


def load_profiles(path: str | Path = DEFAULT_DATASET) -> tuple[Profile, ...]:
    path = Path(path)
    extension = path.suffix.lower()
    if extension not in (".csv", ".jsonl"):
        raise DatasetError("Поддерживаются только .csv и .jsonl")
    result, seen = [], set()

    def add(row, line):
        try:
            profile = Profile.from_mapping(row, csv_value=extension == ".csv")
            if profile.id in seen:
                raise ValueError(f"повторяющийся id {profile.id}")
        except (ValueError, TypeError) as exc:
            raise DatasetError(f"{path.name}, строка {line}: {exc}") from exc
        seen.add(profile.id)
        result.append(profile)

    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            if extension == ".csv":
                reader = csv.DictReader(stream)
                headers = reader.fieldnames or []
                missing = set(Profile.__dataclass_fields__) - set(headers)
                if missing or len(headers) != len(set(headers)):
                    raise DatasetError(f"{path.name}: неверный CSV-заголовок; отсутствуют поля {sorted(missing)}")
                for row in reader:
                    if None in row:
                        raise DatasetError(f"{path.name}, строка {reader.line_num}: лишние значения CSV")
                    add(row, reader.line_num)
            else:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError as exc:
                        raise DatasetError(f"{path.name}, строка {line_no}: некорректный JSON") from exc
                    add(row, line_no)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetError(f"Не удалось прочитать {path.name}: {exc}") from exc
    if not result:
        raise DatasetError(f"{path.name}: датасет пуст")
    return tuple(sorted(result, key=lambda p: p.id))


def convert_csv_to_jsonl(source: str | Path, target: str | Path) -> int:
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise DatasetError("Исходный файл и результат должны различаться")
    profiles = load_profiles(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for profile in profiles:
            stream.write(json.dumps(profile.to_dict(), ensure_ascii=False) + "\n")
    return len(profiles)
