import csv
from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from src import DatasetError, FilterEngine, QueryError, SearchRequest, find_contractors, load_profiles
from src.data_loader import DEFAULT_DATASET, convert_csv_to_jsonl
from src.llm_contract import apply_explanations, build_llm_messages
from src.models import Profile

ROOT = Path(__file__).resolve().parents[1]


def profile(**changes):
    row = dict(id="p1", anon_name="Тестовый подрядчик", categories=["Ведущий"], city="Алматы",
               synthetic=True, city_imputed=False, price_imputed=False, price_from_kzt=100000,
               event_formats=["корпоратив"], languages=["русский", "казахский"],
               max_hours=6, busy_dates=[], description="Проводит двуязычные конференции с 2018 года.")
    row.update(changes)
    return Profile.from_mapping(row)


def query(**changes):
    values = dict(city="Алматы", date="2026-10-10", category="Ведущий",
                  event_format="корпоратив", budget_kzt=200000)
    values.update(changes)
    return SearchRequest(**values)


def run(profiles=None, **changes):
    return FilterEngine(profiles if profiles is not None else [profile()]).search(query(**changes))


def test_all_66_profiles_and_13_synthetic():
    profiles = load_profiles()
    assert len(profiles) == 66
    assert sum(p.synthetic for p in profiles) == 13
    assert len({p.id for p in profiles}) == 66


def test_csv_and_jsonl_identical_including_nulls_and_flags():
    csv_profiles = load_profiles(ROOT / "data/hackathon-dataset-anonymized.csv")
    assert csv_profiles == load_profiles()
    assert sum(p.max_hours is None for p in csv_profiles) == 9
    assert sum(p.city_imputed for p in csv_profiles) == 8
    assert sum(p.price_imputed for p in csv_profiles) == 18


def test_conversion_round_trip(tmp_path):
    output = tmp_path / "copy.jsonl"
    assert convert_csv_to_jsonl(ROOT / "data/hackathon-dataset-anonymized.csv", output) == 66
    assert load_profiles(output) == load_profiles()


@pytest.mark.parametrize("content", ["not-json\n", "[]\n", '{}\n', "\n"])
def test_bad_jsonl_rejected(tmp_path, content):
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError):
        load_profiles(path)


def test_duplicate_ids_rejected_with_location(tmp_path):
    path = tmp_path / "dupes.jsonl"
    line = json.dumps(profile().to_dict())
    path.write_text(line + "\n" + line, encoding="utf-8")
    with pytest.raises(DatasetError, match="строка 2"):
        load_profiles(path)


@pytest.mark.parametrize("change", [
    {"synthetic": "False"}, {"price_from_kzt": -1}, {"price_from_kzt": True},
    {"price_from_kzt": float("nan")}, {"max_hours": 0}, {"max_hours": ""},
    {"categories": "Ведущий"}, {"busy_dates": ["2026-02-30"]},
    {"busy_dates": ["20261010"]}, {"description": ""},
])
def test_bad_profiles_rejected(change):
    with pytest.raises(ValueError):
        profile(**change)


def test_empty_or_corrupt_csv_headers(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,city\np1,Алматы\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="заголовок"):
        load_profiles(path)


@pytest.mark.parametrize("changes,reason", [
    ({"busy_dates": ["2026-10-10"]}, "busy_date"),
    ({"price_from_kzt": 200001}, "over_budget"),
    ({"event_formats": ["свадьба"]}, "event_format"),
    ({"languages": ["английский"]}, "languages"),
    ({"max_hours": 3}, "duration"),
])
def test_each_hard_filter(changes, reason):
    result = run([profile(**changes)], languages=["русский"], duration_hours=4)
    assert result["status"] == "no_matches"
    assert result["cards"] == []
    assert result["excluded"] == [{"id": "p1", "reasons": [reason]}]


def test_city_and_category_are_exact_matches():
    assert run(city="алматы")["status"] == "no_category_in_city"
    assert run(category="Вед")["status"] == "no_category_in_city"
    assert run([profile(city="Астана")])["status"] == "no_category_in_city"


def test_whitespace_is_trimmed_but_no_fuzzy_matching():
    assert run(city=" Алматы ", category=" Ведущий ")["status"] == "ok"


def test_category_membership_for_multicategory_venue():
    p = profile(categories=["Банкетный зал", "Ресторан"])
    assert run([p], category="Ресторан")["status"] == "ok"
    occupied = replace(p, busy_dates=("2026-10-10",))
    assert run([occupied], category="Банкетный зал")["status"] == "no_matches"


def test_budget_and_hours_boundaries_are_inclusive():
    assert run(budget_kzt=100000, duration_hours=6)["status"] == "ok"
    assert run(budget_kzt=99999)["status"] == "no_matches"
    assert run(duration_hours=6.1)["status"] == "no_matches"


def test_null_hours_means_no_presence_requirement():
    result = run([profile(max_hours=None)], duration_hours=12)
    assert result["status"] == "ok"
    assert result["cards"][0]["match_facts"]["duration_not_applicable"] is True


def test_all_languages_required_optional_filters_can_be_absent():
    assert run(languages=["русский", "казахский"])["status"] == "ok"
    assert run(languages=["русский", "английский"])["status"] == "no_matches"
    assert run([profile(languages=["английский"], max_hours=1)])["status"] == "ok"


def test_overlapping_rejections_do_not_claim_unique_totals():
    r = run([profile(price_from_kzt=300000, busy_dates=["2026-10-10"])])
    assert r["summary"]["excluded_profiles"] == 1
    assert sum(r["summary"]["exclusions_by_reason"].values()) == 2
    assert r["summary"]["reason_counts_overlap"] is True
    assert "несколько причин" in r["message"]


def test_b_and_c_are_distinct_and_explained():
    b = run(category="Флорист")
    c = run(budget_kzt=0)
    assert (b["outcome"], c["outcome"]) == ("B", "C")
    assert b["message"] and c["message"]
    assert b["summary"]["in_city_category"] == 0
    assert c["summary"]["in_city_category"] == 1


def test_one_or_two_results_stay_one_or_two():
    for n in (1, 2):
        r = run([profile(id=f"p{i}") for i in range(n)])
        assert len(r["cards"]) == n
        assert "после фильтрации осталось" in r["message"]


def test_top_five_top_three_and_stable_id_tie_break():
    profiles = [profile(id=f"p{i}", price_from_kzt=100000) for i in range(7)]
    first = run(profiles)
    assert [p["id"] for p in first["candidates"]] == ["p0", "p1", "p2", "p3", "p4"]
    assert [p["id"] for p in first["cards"]] == ["p0", "p1", "p2"]
    assert first["summary"]["matched"] == 7
    assert run(list(reversed(profiles))) == first


def test_cheaper_first_then_id():
    r = run([profile(id="b", price_from_kzt=90000), profile(id="a"), profile(id="c", price_from_kzt=80000)])
    assert [p["id"] for p in r["cards"]] == ["c", "b", "a"]


def test_flags_never_dropped_or_used_as_quality_ranking():
    r = run([profile(synthetic=False, city_imputed=True, price_imputed=True)])
    assert r["cards"][0]["synthetic"] is False
    assert r["cards"][0]["price_imputed"] is True
    assert r["cards"][0]["city_imputed"] is True
    assert "цена дополнена" in r["cards"][0]["explanation"]


def test_fallback_prefers_concrete_fact_over_greeting():
    p = profile(description="Приветствую всех, дорогие друзья! Провёл 300 корпоративов. До встречи!")
    text = run([p])["cards"][0]["explanation"]
    assert "Провёл 300 корпоративов" in text
    assert "Приветствую" not in text


@pytest.mark.parametrize("change", [
    {"date": "2026-09-22"}, {"date": "2027-01-01"}, {"date": "2026-11-31"},
    {"date": "20261010"}, {"budget_kzt": -1}, {"budget_kzt": True},
    {"budget_kzt": "100000"}, {"budget_kzt": float("inf")}, {"budget_kzt": 0.5},
    {"duration_hours": 0}, {"duration_hours": True}, {"duration_hours": float("nan")},
    {"city": " "}, {"languages": "русский"}, {"languages": [123]},
])
def test_bad_requests_rejected(change):
    with pytest.raises(QueryError):
        query(**change)


def test_calendar_edges_are_allowed():
    assert run(date="2026-09-23")["status"] == "ok"
    assert run(date="2026-12-31")["status"] == "ok"


def test_unexpected_request_fields_rejected():
    with pytest.raises(QueryError):
        SearchRequest.from_mapping({**query().to_dict(), "budjet": 100})


def test_public_function_accepts_plain_dicts_and_does_not_mutate():
    rows, q = [profile().to_dict()], query().to_dict()
    original = json.dumps([rows, q], sort_keys=True)
    assert find_contractors(rows, q)["status"] == "ok"
    assert json.dumps([rows, q], sort_keys=True) == original


def test_same_real_request_has_same_full_response_and_input_order_irrelevant():
    profiles = list(load_profiles())
    first = run(profiles, budget_kzt=1500000)
    random.Random(42).shuffle(profiles)
    assert run(profiles, budget_kzt=1500000) == first


def test_date_change_changes_cards_and_busy_ids_are_excluded():
    e = FilterEngine(load_profiles())
    a = e.search(query(budget_kzt=1500000))
    b = e.search(query(budget_kzt=1500000, date="2026-10-11"))
    assert [p["id"] for p in a["cards"]] != [p["id"] for p in b["cards"]]
    for r in (a, b):
        for p in r["candidates"]:
            assert r["request"]["date"] not in p["busy_dates"]
        for card in r["cards"]:
            assert r["request"]["date"] in card["explanation"]


def test_real_dataset_against_independent_filter_for_200_queries():
    e = FilterEngine(load_profiles())
    rng = random.Random(79)
    options = e.options()
    for _ in range(200):
        q = query(city=rng.choice(options["cities"]), category=rng.choice(options["categories"]),
                  date=rng.choice(["2026-09-23", "2026-10-10", "2026-11-14", "2026-12-26"]),
                  budget_kzt=rng.choice([0, 150000, 500000, 1500000, 10000000]),
                  event_format=rng.choice(options["event_formats"]),
                  languages=rng.choice([[], ["русский"], ["русский", "казахский"]]),
                  duration_hours=rng.choice([None, 2, 6, 12]))
        expected = [p for p in e.profiles
                    if p.city == q.city and q.category in p.categories
                    and q.date not in p.busy_dates and p.price_from_kzt <= q.budget_kzt
                    and q.event_format in p.event_formats
                    and all(lang in p.languages for lang in q.languages)
                    and (q.duration_hours is None or p.max_hours is None or p.max_hours >= q.duration_hours)]
        expected.sort(key=lambda p: (p.price_from_kzt, p.id))
        actual = e.search(q)
        assert actual["summary"]["matched"] == len(expected)
        assert [p["id"] for p in actual["candidates"]] == [p.id for p in expected[:5]]


def test_llm_contract_preserves_order_and_does_not_modify_original():
    r = run([profile(id="a"), profile(id="b")])
    updated = apply_explanations(r, {"b": "Текст B.", "a": "Текст A."})
    assert [c["id"] for c in updated["cards"]] == ["a", "b"]
    assert updated["cards"][0]["explanation"] == "Текст A."
    assert r["cards"][0]["explanation_source"] == "deterministic_template"
    assert json.loads(build_llm_messages(r)[1]["content"]) == r["llm_context"]


@pytest.mark.parametrize("texts", [{}, {"wrong-id": "Текст"}, {"p1": ""}, {"p1": 12}])
def test_llm_bad_ids_or_texts_rejected(texts):
    with pytest.raises(ValueError):
        apply_explanations(run(), texts)


def test_cli_reads_request_and_writes_valid_json(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(query().to_dict(), ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "output.json"
    execution = subprocess.run([sys.executable, str(ROOT / "main.py"), "search", "--request", str(path),
                                "--output", str(output)], cwd=tmp_path, capture_output=True)
    assert execution.returncode == 0, execution.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["request"] == query().to_dict()


def test_cli_bad_request_has_nonzero_exit(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"date":"bad"}', encoding="utf-8")
    execution = subprocess.run([sys.executable, str(ROOT / "main.py"), "search", "--request", str(path)], capture_output=True)
    assert execution.returncode == 2
    assert "error" in json.loads(execution.stderr)
