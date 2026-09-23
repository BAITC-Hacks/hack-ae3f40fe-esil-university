"""Воспроизводимые сценарии на исходных 66 профилях, без ключей API."""

import json
from pathlib import Path
import sys
from src import FilterEngine, load_profiles


def main():
    engine = FilterEngine(load_profiles())
    for path in sorted((Path(__file__).parent / "examples").glob("*_request.json")):
        result = engine.search(json.loads(path.read_text(encoding="utf-8")))
        print(f"\n{path.stem}: {result['status']}")
        print(result["message"])
        for card in result["cards"]:
            label = "синтетический" if card["synthetic"] else "исходный анонимизированный"
            print(f"  {card['id']} | {card['anon_name']} | от {card['price_from_kzt']} ₸ | {label}")
            print("  " + card["explanation"])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
