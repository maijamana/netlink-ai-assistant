# NetLink AI Assistant — RAG & Evaluation

Прототип RAG-асистента служби підтримки для інтернет-провайдера **NetLink**.

Система:

- знаходить релевантні статті в knowledge base;
- генерує grounded-відповіді українською мовою;
- повертає ID використаних джерел;
- утримується від відповіді, якщо knowledge base не містить достатньо інформації;
- містить retrieval та generation evaluation із regression-порівнянням v1/v2.

## Архітектура

```text
Knowledge Base
    ↓
title + article text
    ↓
Multilingual E5 embeddings
    ↓
FAISS exact search
    ↓
Top-3 retrieved articles
    ↓
Grounded LLM generation
    ↓
Structured output:
answerable / answer / sources
    ↓
Application-level validation
```

Retrieval і generation оцінюються окремо, щоб відрізняти retrieval failures
від помилок answerability, generation або citation.

## Retrieval

Knowledge base містить 11 коротких тематичних статей, тому використовується:

> **1 article = 1 chunk**

Додатковий fixed-size chunking створив би зайву складність і міг би розділити
пов’язані факти. Для довших production-документів доцільнішими були б
structure-aware або recursive chunking з evaluation розміру chunk та overlap.

Retriever використовує:

- `intfloat/multilingual-e5-small`;
- префікси `query:` / `passage:`;
- L2-normalized embeddings;
- `FAISS IndexFlatIP`;
- `top_k = 3`.

Для L2-нормалізованих векторів inner product еквівалентний cosine similarity,
тому індекс виконує exact cosine ranking.

Для 11 документів approximate indexes на зразок HNSW не потрібні. Для більшого
production-корпусу можна було б розглянути pgvector, Qdrant або Azure AI Search
залежно від вимог до filtering, persistence, update та scaling.

## Out-of-KB Handling

Fixed similarity threshold виявився недостатньо надійним для answerability.

Наприклад, out-of-KB запит про кабельне телебачення отримав високий embedding
similarity score, хоча жодна KB-стаття не містила потрібної інформації. Тому
retrieval scores використовуються як ranking signals, а не як calibrated
confidence scores.

Generation layer отримує top-3 candidate articles і визначає, чи містить
контекст достатньо інформації для відповіді.

Якщо ні:

```json
{
  "answerable": false,
  "answer": "У базі знань немає достатньої інформації, щоб відповісти на це запитання. Будь ласка, зверніться до оператора.",
  "sources": []
}
```

Після цього application повертає deterministic fallback із рекомендацією
звернутися до оператора.

Generated source IDs також перевіряються за фактично retrieved документами.

## Dialogue Flow

Для сценарію «не працює інтернет» реалізовано stateful troubleshooting flow.

Flow послідовно збирає інформацію про:

- стан індикаторів роутера;
- наявність масової аварії;
- стан оплати;
- спробу перезавантаження;
- результат troubleshooting.

Основна логіка:

```text
START
  ↓
ISSUE
  ↓
CHECK INDICATORS
  ├── LOS red → ESCALATE
  ├── WAN problem → CHECK OUTAGE
  └── green → CHECK PAYMENT
                     ↓
                 CHECK REBOOT
                  ├── no → RAG instructions
                  └── yes
                        ↓
                    RESOLVED?
                    ├── yes → DONE
                    └── no → ESCALATE
```

Business-critical branching контролюється deterministic state machine, а RAG
використовується для отримання інформації з knowledge base там, де це доречно.

Наприклад, якщо під час troubleshooting користувач ставить стороннє питання
(наприклад, про вартість тарифу), система відповідає через RAG і після цього
повертається до незавершеного кроку діагностики.

### Escalation conditions

Ескалація відбувається, якщо:

- горить червоний LOS;
- базові troubleshooting steps виконано, але інтернет не відновився.

Якщо підтверджено масову аварію, індивідуальна заявка не створюється.

## Evaluation

Golden dataset розширено до **18 запитань**, серед яких:

- single-source factual questions;
- paraphrases;
- multi-source questions;
- схожі або потенційно плутані теми;
- out-of-KB cases.

### Retrieval metrics

Retrieval оцінюється на 15 answerable-запитаннях:

| Metric | Result |
| ------ | -----: |
| Hit@1  |  0.933 |
| Hit@3  |  1.000 |
| MRR    |  0.967 |

`Hit@K` тут означає, що хоча б одне expected source знайдено в top-k.
Для multi-source cases evaluator також рахує `Source Recall@3` — частку всіх
expected sources, знайдених у top-3.

Єдиний Hit@1 failure усе одно містив правильне джерело на rank 2, що підтримує
використання top-3 context для generation.

### Generation metrics

Generation оцінюється через deterministic checks та LLM-as-judge.
Для generation та judge використовується `gpt-4.1-mini`.

Deterministic metrics:

- Answerability Accuracy;
- Citation Recall;
- Citation Precision.

LLM-as-judge metrics:

- Correctness;
- Faithfulness.

Judge виставляє кожну оцінку за шкалою 0–2. У фінальній comparison table ці
значення нормалізуються до діапазону 0–1.

Обидва підходи використовуються навмисно: semantic judging не перевіряє
повністю structured behavior системи.

Наприклад, у v1 case G17 про Netflix або телебачення в тарифах містив семантично
прийнятний текст, але помилково повертав `answerable=true`. Deterministic
метрика виявила цю помилку, тоді як LLM judge — ні. Окремий cable-TV case G09
був правильно класифікований як out-of-KB уже у v1.

## Regression Experiment

Дві версії generation prompt оцінено на тому самому fixed golden dataset.

Між v1 та v2 не змінювалися knowledge base, embedding model, retrieval strategy,
`top_k` або generation model — лише prompt. Це дозволяє розглядати порівняння
як контрольований prompt regression experiment.

### v1 — Grounded generation baseline

Базова версія prompt із правилами:

- відповідати лише на основі retrieved context;
- не використовувати зовнішні знання;
- повертати structured output (`answerable`, `answer`, `sources`);
- використовувати лише retrieved source IDs.

### v2 — Strict answerability

У v2 змінено лише generation prompt.

Додано строге правило: тематично пов’язаний retrieved документ ще не означає,
що запит є answerable. `answerable=true` дозволено лише тоді, коли context явно
містить інформацію, необхідну для відповіді на конкретне запитання.

Також явно заборонено трактувати відсутність інформації як доказ того, що певної
послуги або умови немає.

| Version | Hit@1 | Hit@3 | MRR | Retrieval Source Recall@3 | Answerability | Citation Recall | Citation Precision | Correctness | Faithfulness |
| ------- | ----: | ----: | --: | ------------------------: | ------------: | --------------: | -----------------: | ----------: | -----------: |
| v1 | 0.933 | 1.000 | 0.967 | 0.967 | 0.944 | 0.917 | 0.944 | 0.972 | 1.000 |
| v2 | 0.933 | 1.000 | 0.967 | 0.967 | **1.000** | **0.972** | **0.963** | **1.000** | **1.000** |

Retrieval metrics однакові, оскільки retrieval навмисно не змінювався.
Фінальна конфігурація — **v2**.

Покращення стосується aggregate metrics і не означає perfect attribution у
кожному окремому case: у v2 залишаються приклади з неповним або надлишковим
набором sources.

## «Good Enough» Threshold

Для цього прототипу я вважаю retrieval достатнім, якщо:

```text
Hit@3 >= 0.95
```

оскільки generation використовує top-3 retrieved passages.

Для generation пріоритети такі:

1. коректний out-of-KB abstention;
2. faithfulness;
3. correctness;
4. source attribution.

Для customer support безпечне утримання від відповіді краще за непідтверджену
політику компанії або рекомендацію.

Поточний evaluation set невеликий, тому результати слід розглядати як
prototype-level evidence, а не production-quality validation.

## Обмеження та наступні кроки

Поточні обмеження:

- лише 18 evaluation questions;
- повторне tuning на тому самому dataset створює ризик overfitting;
- використовується лише dense retrieval;
- LLM-as-judge може давати нестабільні оцінки;
- немає production-інфраструктури та observability.

За наявності додаткового часу варто було б:

- розділити tuning і holdout test sets;
- розширити evaluation реалістичними customer queries;
- перевірити hybrid lexical + dense retrieval і reranking;
- відкалібрувати out-of-KB detection;
- виміряти latency та API cost;
- додати tracing і production monitoring.

## Запуск

Prerequisites: Python 3.8 або новіший. Рекомендовано запускати всі команди з
кореня repository. Під час першого retrieval-запуску модель
`intfloat/multilingual-e5-small` завантажується з Hugging Face.

Створіть virtual environment і встановіть залежності:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Створіть `.env` із template:

```bash
cp .env.example .env
```

Після цього додайте свій `OPENAI_API_KEY` у `.env`.

Запустіть асистента:

```bash
python main.py
```

`main.py` використовує фінальну prompt-конфігурацію v2.

Запустити interactive dialogue demo:

```bash
python dialogue_demo.py
```

Запустіть retrieval evaluation:

```bash
python -m evaluation.evaluate_retrieval \
  | tee results/retrieval/evaluation.txt
```

Команда створює `results/retrieval/evaluation.json`, який використовує
фінальне порівняння.

За потреби запустіть manual retrieval smoke test:

```bash
python -m evaluation.manual_retrieval_test \
  | tee results/retrieval/manual_queries.txt
```

Відтворіть generation evaluation та judge для обох prompt versions:

```bash
python -m evaluation.evaluate_generation --version v1 \
  | tee results/generation/v1/evaluation.txt
python -m evaluation.judge_generation --version v1 \
  | tee results/generation/v1/judge.txt

python -m evaluation.evaluate_generation --version v2 \
  | tee results/generation/v2/evaluation.txt
python -m evaluation.judge_generation --version v2 \
  | tee results/generation/v2/judge.txt
```

Побудуйте фінальне порівняння:

```bash
python -m evaluation.compare_versions \
  | tee results/comparison/final_comparison.txt
```

Основні generated artifacts:

```text
results/
├── retrieval/
│   ├── evaluation.txt
│   ├── evaluation.json
│   └── manual_queries.txt
├── generation/
│   ├── v1/
│   │   ├── evaluation.txt
│   │   ├── evaluation.json
│   │   ├── judge.txt
│   │   └── judged.json
│   └── v2/
│       ├── evaluation.txt
│       ├── evaluation.json
│       ├── judge.txt
│       └── judged.json
└── comparison/
    ├── final_comparison.txt
    └── evaluation_results.csv
```
