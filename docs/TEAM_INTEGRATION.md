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
| explanation_status | Только в общем API: источник текста, причина резервного ответа, провайдер |

Полный JSON — `examples/01_dense_response.json`. `cards` уже имеют простые фактические подписи с `explanation_source=deterministic_template`; это резерв, а не заявленная работа нейросети.

## Готовая интеграция участников 1 и 2

`POST /recommendations` вызывает общий сервис `src/recommendation_service.py`:

1. `FilterEngine` проверяет запрос, применяет фильтры и задаёт порядок карточек.
2. При B/C возвращает точное `message` и счётчики без сети.
3. При A передаёт `LLMExplainer` первые три выбранных профиля и статус `SUCCESS`.
4. Добавляет совместимые имена полей: `name=anon_name`, `budget=budget_kzt`, `language=languages`, `duration=duration_hours`.
5. Ждёт ответ не больше 8 секунд. При ошибке сохраняет исходные карточки и фактические подписи.
6. Допустимые объяснения подставляет по ID, сохраняя все остальные поля.

Выбор провайдера: `LLM_PROVIDER=openai` либо `nvidia`; настройки находятся в `.env.example`. API читает локальный `.env` при старте и не перезаписывает системные переменные. В одном поиске используется один провайдер, без сетевых повторов. SDK установлен через `requirements.txt`.

В `explanation_status.source` будет `external_llm` для ответа модели или `deterministic_template` для местного текста. Значения `reason`: `generated`, `empty_result`, `missing_api_key`, `sdk_unavailable`, `client_initialization_failed`, `request_failed`, `invalid_response`, `unverified_source`, `timeout`. Не показывайте локальные подписи пользователю как работу нейросети.

В модуле участника 2 добавлены поля `source` и `fallback_reason`; существующие ключи ответа сохранены. Кеш теперь отдельный для каждого экземпляра, защищён при параллельных запросах и хранит только успешные ответы модели. Одинаковый запрос после перезапуска может дать другой текст; отбор и порядок детерминированы всегда.

Проверки формы, ID и клише не доказывают истинность каждого утверждения. Для защиты проекта дополнительно просмотрите реальные объяснения после настройки ключа. Имитация HTTP не подтверждает доступность конкретной модели в аккаунте.

## Участнику 3: подключение интерфейса

На момент проверки `app.py` отсутствует в `main`, `feature/backend-filtering` и `docs/readme`. Поэтому полный пользовательский путь Streamlit пока не проверен. Тестовая страница API доступна по `http://127.0.0.1:8000/docs`.

Рекомендуемый вариант для Streamlit — общий HTTP API. Пример кода для будущего `app.py`:

```python
import httpx
import streamlit as st

# user_request — словарь полей формы по таблице выше.
try:
    response = httpx.post(
        "http://127.0.0.1:8000/recommendations", json=user_request, timeout=12.0
    )
    response.raise_for_status()
    result = response.json()
    st.write(result["message"])
    for card in result["cards"]:
        st.subheader(card["anon_name"])
        st.write(f"От {card['price_from_kzt']:,} ₸")
        st.caption("Синтетический профиль" if card["synthetic"]
                   else "Исходный анонимизированный профиль")
        st.write(card["explanation"])
except httpx.HTTPStatusError as exc:
    st.error(f"Проверьте параметры запроса: {exc.response.status_code}")
except httpx.RequestError:
    st.error("Сервер подбора недоступен. Проверьте, запущен ли API.")
```

Для этого примера участник 3 должен добавить `streamlit` и `httpx` в зависимости интерфейса. API запускается отдельно командой `python -m uvicorn api:app --reload`. Значения списков и границы календаря берите из `GET /options`.

Общий сервис можно вызвать из асинхронного Python-кода напрямую:

```python
from src import FilterEngine, load_profiles
from src.recommendation_service import RecommendationService

service = RecommendationService(FilterEngine(load_profiles()))
# Внутри async-функции:
# result = await service.recommend(user_request)
```

При прямом импорте задайте переменные среды самостоятельно: `.env` загружает именно `api.py`. Вызов `engine.search(user_request)` запускает только фильтры и местные подписи, без внешней модели.

Показывайте `result.message` при любом исходе, особенно если карточек меньше трёх. Сохраняйте метки `city_imputed` и `price_imputed`. Цена указывается «от»; отсутствие даты в `busy_dates` не означает подтверждённое бронирование. Для длительности без ограничения передавайте `null`, а не 0; все выбранные языки обязательны.

## Проверка и объединение

Запуск: `python -m pip install -r requirements-dev.txt`, затем `python -m pytest -q`. Тесты удаляют настоящие ключи из своего окружения, используют имитацию HTTP и не обращаются к провайдерам. Результаты: `docs/VALIDATION.md`.

Изменения находятся в PR #2, ветка `feature/backend-filtering`. Основная ветка получит их после принятия PR командой. Корневой README команды сохранён; инструкция по запуску находится в `BACKEND_README.md`. Отдельный PR #1 содержит README участника 3, но самого интерфейса в нём пока нет.
