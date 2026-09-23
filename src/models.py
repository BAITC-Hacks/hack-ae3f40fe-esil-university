"""Общий контракт. Ядро использует только стандартную библиотеку Python."""

from dataclasses import asdict, dataclass
from datetime import date
import math
from typing import Any, Mapping

CALENDAR_START = "2026-09-23"
CALENDAR_END = "2026-12-31"
LIST_FIELDS = ("categories", "event_formats", "languages", "busy_dates")
FLAG_FIELDS = ("synthetic", "city_imputed", "price_imputed")


class QueryError(ValueError):
    """Неверные параметры пользовательского запроса."""


def text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: требуется непустая строка")
    return value.strip()


def iso_date(value: Any, field: str) -> str:
    value = text(value, field)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field}: нужна существующая дата YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field}: используйте формат YYYY-MM-DD")
    return value


def number(value: Any, field: str, *, positive: bool = False,
           integer: bool = False, csv_value: bool = False) -> int | float:
    if csv_value and isinstance(value, str):
        try:
            value = float(value)
        except ValueError as exc:
            raise ValueError(f"{field}: требуется число") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}: требуется число, а не строка или bool")
    if (isinstance(value, float) and not math.isfinite(value)) or value < 0 or (positive and value == 0):
        raise ValueError(f"{field}: недопустимое значение")
    if integer and int(value) != value:
        raise ValueError(f"{field}: требуется целое число")
    return int(value) if integer or int(value) == value else float(value)


def string_list(value: Any, field: str, *, csv_value: bool = False,
                allow_empty: bool = False) -> tuple[str, ...]:
    if csv_value and isinstance(value, str):
        value = value.split("|") if value.strip() else []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field}: требуется список строк")
    values = tuple(sorted(set(text(v, field) for v in value)))
    if not values and not allow_empty:
        raise ValueError(f"{field}: список не должен быть пустым")
    return values


def flag(value: Any, field: str, *, csv_value: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if csv_value and isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"{field}: требуется true или false")


@dataclass(frozen=True)
class Profile:
    id: str
    anon_name: str
    categories: tuple[str, ...]
    city: str
    city_imputed: bool
    synthetic: bool
    price_from_kzt: int
    price_imputed: bool
    event_formats: tuple[str, ...]
    languages: tuple[str, ...]
    max_hours: int | float | None
    busy_dates: tuple[str, ...]
    description: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, csv_value: bool = False) -> "Profile":
        if not isinstance(row, Mapping):
            raise ValueError("профиль должен быть JSON-объектом")
        missing = set(cls.__dataclass_fields__) - set(row)
        if missing:
            raise ValueError("отсутствуют поля: " + ", ".join(sorted(missing)))
        values = {k: text(row[k], k) for k in ("id", "anon_name", "city", "description")}
        for k in LIST_FIELDS:
            values[k] = string_list(row[k], k, csv_value=csv_value, allow_empty=k == "busy_dates")
        values["busy_dates"] = tuple(iso_date(d, "busy_dates") for d in values["busy_dates"])
        for k in FLAG_FIELDS:
            values[k] = flag(row[k], k, csv_value=csv_value)
        values["price_from_kzt"] = number(row["price_from_kzt"], "price_from_kzt",
                                          integer=True, csv_value=csv_value)
        hours = row["max_hours"]
        if csv_value and isinstance(hours, str) and not hours.strip():
            hours = None
        values["max_hours"] = None if hours is None else number(
            hours, "max_hours", positive=True, csv_value=csv_value)
        return cls(**values)

    def to_dict(self) -> dict:
        result = asdict(self)
        for field in LIST_FIELDS:
            result[field] = list(result[field])
        return result


@dataclass(frozen=True)
class SearchRequest:
    city: str
    date: str
    event_format: str
    category: str
    budget_kzt: int
    languages: tuple[str, ...] = ()
    duration_hours: int | float | None = None

    def __post_init__(self):
        try:
            for field in ("city", "event_format", "category"):
                object.__setattr__(self, field, text(getattr(self, field), field))
            value = iso_date(self.date, "date")
            if not CALENDAR_START <= value <= CALENDAR_END:
                raise ValueError(f"Календарь доступен только с {CALENDAR_START} по {CALENDAR_END}")
            object.__setattr__(self, "date", value)
            object.__setattr__(self, "budget_kzt", number(self.budget_kzt, "budget_kzt", integer=True))
            object.__setattr__(self, "languages", string_list(self.languages, "languages", allow_empty=True))
            if self.duration_hours is not None:
                object.__setattr__(self, "duration_hours", number(self.duration_hours, "duration_hours", positive=True))
        except ValueError as exc:
            raise QueryError(str(exc)) from exc

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SearchRequest":
        if not isinstance(values, Mapping):
            raise QueryError("Запрос должен быть JSON-объектом")
        try:
            return cls(**values)
        except TypeError as exc:
            raise QueryError("Проверьте поля запроса: city, date, event_format, category, "
                             "budget_kzt, languages, duration_hours") from exc

    def to_dict(self) -> dict:
        result = asdict(self)
        result["languages"] = list(self.languages)
        return result
