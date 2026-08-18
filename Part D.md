# Частина D — Письмові відповіді

## 1. Який LLM обрано і чому?

Для generation у прототипі використано `gpt-4.1-mini`.

Основним критерієм вибору був компроміс між quality, latency та cost. Для цього завдання модель не має виконувати складний reasoning, бо основна інформація приходить через retrieval, а generation має коректно визначити answerability, сформувати коротку grounded-відповідь і повернути джерела.

Використання більшої моделі дало б більшу вартість і можливо більшу latency без сильної користі для невеликого сценарію.

Якість контролюється не лише вибором моделі, а і архітектурою:

- top-3 retrieval;
- строгі grounding rules;
- structured output (`answerable`, `answer`, `sources`);
- application-level validation;
- окрема evaluation для correctness, faithfulness та answerability.

На нашому golden dataset фінальна конфігурація v2 отримала `1.000` за Answerability Accuracy, normalized Correctness та Faithfulness.

Для production я додатково вимірювала б p50/p95 latency, cost per request і якість на значно більшому наборі реальних запитів перед остаточним вибором моделі.

## 2. Що зміниться для повністю on-prem / self-hosted рішення?

Основна архітектура RAG залишилася б схожою, але всі зовнішні API-компоненти потрібно було б замінити локальними.

Теперішня embedding model `multilingual-e5-small` працює локально, тому retrieval layer зазнав би мінімальних змін.

Основна заміна стосувалася б generation LLM. Замість зовнішнього API використовувалася б self-hosted instruction model через, наприклад, vLLM.

Архітектура виглядала б приблизно так:

`User → application → retriever → local vector store → local LLM inference server`

Для production також знадобилися б:

- GPU infrastructure та capacity planning;
- model serving і autoscaling;
- batching запитів;
- контроль memory/VRAM;
- локальне зберігання моделей;
- monitoring latency, throughput і GPU utilization;
- механізми rolling update моделей.

Замість локального FAISS для більшого knowledge base я б використала наприклад pgvector або Qdrant.

Головний trade-off self-hosted підходу це більший контроль над даними та відсутність зовнішнього API, але значно вища операційна складністб.

## 3. Де поточне рішення зламається першим під реальним навантаженням / реальними користувачами?

Думаю, першою проблемою буде **якість на реальних запитах і generation latency**.

Поточний knowledge base містить тільки 11 добре структурованих статей, а golden dataset 18 запитів. У реальному контакт-центрі з’являться помилки, розмовна мова, неоднозначні питання, кілька проблем в одному повідомленні тощо. 

Тому перший ризик - падіння retrieval recall та неправильне визначення answerability.

Друга проблема — зовнішній LLM API. При збільшенні concurrency зростатимуть latency, cost і ймовірність rate-limit/timeout помилок.

Для production я б додала:

- більший evaluation set із реальних support-запитів;
- tracing retrieval та generation кроків;
- p50/p95 latency і error-rate monitoring;
- retries з backoff для transient API failures;
- rate limiting;
- caching для повторюваних запитів;
- чітку ескалацію до оператора при низькій впевненості.

Тобто поточне рішення є достатнім як перевірюваний RAG-прототип, але перед production основний фокус був би на evaluation, observability, reliability та масштабуванні inference.