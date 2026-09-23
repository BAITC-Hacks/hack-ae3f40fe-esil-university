"""Запуск фильтров из терминала: Python 3.10+, сторонние пакеты не нужны."""

import argparse
import json
from pathlib import Path
import sys

from src import DatasetError, FilterEngine, QueryError, SearchRequest, load_profiles
from src.data_loader import DEFAULT_DATASET, convert_csv_to_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Подбор event-подрядчиков — бэкенд участника 1")
    actions = parser.add_subparsers(dest="command", required=True)
    for name in ("search", "options", "validate"):
        command = actions.add_parser(name)
        command.add_argument("--data", type=Path, default=DEFAULT_DATASET)
        if name == "search":
            command.add_argument("--request", type=Path, required=True, help="JSON-файл запроса")
            command.add_argument("--output", type=Path, help="Записать полный результат в UTF-8")
    converter = actions.add_parser("convert")
    converter.add_argument("--source", type=Path, required=True)
    converter.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "convert":
            count = convert_csv_to_jsonl(args.source, args.target)
            result = {"converted_profiles": count, "output": str(args.target)}
        else:
            engine = FilterEngine(load_profiles(args.data))
            if args.command in ("options", "validate"):
                result = engine.options()
            else:
                query = SearchRequest.from_mapping(json.loads(args.request.read_text(encoding="utf-8-sig")))
                result = engine.search(query)
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if getattr(args, "output", None):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
        else:
            print(serialized)
        return 0
    except (DatasetError, QueryError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
