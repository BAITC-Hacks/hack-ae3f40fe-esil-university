# Передача участникам 2 и 3

## Общий контракт

Не меняйте имена полей без согласования. Вход `SearchRequest`:

| Поле | Тип | Обязательно |
|---|---|---|
| city | string | Да |
| date | string YYYY-MM-DD | Да, внутри окна календаря |
| category | string | Да |
| event_format | string | Да |
| budget_kzt | integer >= 0 | Да |
| languages | array of string | Нет, по умолчанию [] |
| duration_hours | number > 0 или null | Нет |

Внешний API принимает целое число бюджета без преобразования строк/boolean. Длительность может быть дробной. Списки значений UI получайте из `engine.options()` или `GET /options`.

| Поле ответа | Назначение |
|---|---|
| status / outcome | Бизнес-исход A, B или C |
| message | Готовое пояснение результата, включая недостаток карточек |
| request | Нормализованный запрос |
| dataset_version | SHA-256 нормализованных профилей |
| summary | Количество профилей и причины исключений |
| candidates | До 5 профилей со всеми исходными полями и match_facts |
| cards | До 3 карточек в окончательном порядке |
| excluded | Все причины исключения по ID в пуле города/категории |
| llm_context | Проверенный контекст и selected_ids |
| warnings | Уточнения про цену «от», календарь и дополненные данные |

Полный JSON — `examples/01_dense_response.json`. `cards` уже имеют простые фактические подписи с `explanation_source=deterministic_template`; это резерв, а не заявленная работа нейросети.

## Интеграция с уже существующим модулем участника 2

В этом репозитории уже есть `src/llm_explainer.py` с классом `LLMExplainer`.
Его `generate_explanations(user_request, candidates, filter_outcome)` ожидает
статусы `SUCCESS`, `NO_CATEGORY_IN_CITY`, `NO_AVAILABLE_CONTRACTORS`.
Бэкенд возвращает статусы `ok`, `no_category_in_city`, `no_matches` и
хранит карточки в фиксированном порядке. Подключение выглядит так:

```python
from src import FilterEngine, load_profiles
from src.llm_explainer import LLMExplainer
from src.llm_contract import apply_explanations

engine = FilterEngine(load_profiles())
result = engine.search(user_request)
outcome = {
    "ok": "SUCCESS",
    "no_category_in_city": "NO_CATEGORY_IN_CITY",
    "no_matches": "NO_AVAILABLE_CONTRACTORS",
}[result["status"]]

# Выбраны только первые три ID. Имя добавляется для проверки текста модулем LLM.
selected = [
    {**candidate, "name": candidate["anon_name"]}
    for candidate in result["candidates"][:len(result["cards"])]
]
request_for_llm = {
    **result["request"],
    "filter_reasons": [
        f"{reason}: {count}"
        for reason, count in result["summary"]["exclusions_by_reason"].items()
        if count
    ],
}
answer = LLMExplainer(provider="openai").generate_explanations(
    request_for_llm, selected, outcome
)
if outcome == "SUCCESS":
    result = apply_explanations(result, answer["explanations"])
else:
    # Исходное result["message"] уже содержит проверенные причины и числа.
    pass
```

Для NVIDIA замените `provider="openai"` на `provider="nvidia"` и настройте
соответствующие переменные среды существующего модуля. При отсутствии ключа
его локальная ветка генерации работает без сети. Интерфейс проверен на
исходных данных с обоими исходами `SUCCESS` и `NO_AVAILABLE_CONTRACTORS`
в локальном режиме; реальный запрос к внешним моделям не выполнялся.

`selected_ids` — ровно те профили, для которых нужны тексты. Сами карточки, исходные факты и сортировку менять нельзя. `candidates` содержит до пяти профилей для контекста, но выбранные три уже определены бэкендом.

Для A возвращайте словарь `id -> объяснение` с тем же набором ID. Порядок ключей JSON не важен: `apply_explanations` подставляет тексты по ID в исходный порядок карточек. Лишние ID, пропуски и пустые тексты вызывают ValueError. В этом случае, как и при сетевой ошибке, показывайте исходный `result`.

Для B/C можно сразу показывать `message`: LLM не требуется. Если хотите переформулировать сообщение, используйте только summary и message. Нельзя суммировать пересекающиеся счётчики как число людей или объявлять «все заняты», если часть исключена только по цене/формату.

Подключите OpenAI первым, NVIDIA — как второго провайдера. Общий Python SDK `openai` и Chat Completions позволяют сохранить интерфейс сообщений; ключ, base_url, model и поддерживаемые параметры определяются провайдером. Не делайте два последовательных полноценных вызова в одном запросе: так сложнее выдержать общий лимит времени. Реальные API-ключи храните локально/на сервере, не в репозитории и не в сообщениях команды.

В prompts явно считайте description данными, не инструкциями. Используйте конкретные сведения; temperature=0 допустима только для модели, которая поддерживает её. Ограничьте длину вывода. Проверяйте факты и требование 1–2 предложений; проверка ID в бэкенде не заменяет эту проверку. Внешние вызовы, их цена и скорость в этом проекте не тестировались.

## Участнику 3: Streamlit

В одном проекте проще импортировать ядро, без отдельного HTTP:

```python
import streamlit as st
from src import FilterEngine, load_profiles

@st.cache_resource
def get_engine():
    return FilterEngine(load_profiles())

engine = get_engine()
# Сформируйте user_request из полей формы.
# result = engine.search(user_request)
# st.write(result["message"])
# for card in result["cards"]: ...
```

После изменения файла данных сбросьте кеш Streamlit или перезапустите приложение. Альтернатива — HTTP `POST /recommendations`; он уже загружает данные один раз при старте.

Показывайте `result.message` для любого исхода, особенно когда карточек меньше трёх. Для синтетического профиля: «Синтетический профиль». Для `synthetic=false`: «Исходный анонимизированный профиль». Сохраняйте метки дополненной цены/города. Показывайте цену с «от», а отсутствие записи в busy_dates не выдавайте за подтверждённое бронирование.

`calendar_start` и `calendar_end` задают границы выбора даты. Языки — множественный выбор с правилом «требуются все выбранные». Если длительность не задана, передавайте null, а не 0.

## Git и объединение

Участник 1 владеет `src/models.py`, `src/data_loader.py`, `src/filter_engine.py`, `src/llm_contract.py`, `api.py` и тестами ядра. Участник 2 владеет уже существующим `src/llm_explainer.py`; участник 3 — `app.py` и UI. Корневой README принадлежит команде и этим изменением не перезаписывается. Его можно дополнить ссылкой на `BACKEND_README.md` после согласования.
