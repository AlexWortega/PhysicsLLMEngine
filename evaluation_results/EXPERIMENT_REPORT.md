# Physics LLM - Отчёт по экспериментам

## Цель проекта
Обучить языковую модель (100M-500M параметров) предсказывать 2D rigid body физику — позиции, скорости и столкновения объектов на несколько кадров вперёд.

## Архитектура проекта

### Данные
- **Формат**: JSONL с гибридным представлением (текст + структурированные числа)
- **Объём**: 100K сцен (80K train, 10K val, 10K test)
- **Сцены**: 10-50 объектов (круги, прямоугольники) с разными материалами
- **Симуляция**: Pymunk, 200 кадров на сцену, timestep 1/60s
- **Общий размер**: ~133 GB

### Модели

#### 1. LFM2-350M Fine-tuned (работает)
- **База**: LiquidAI/LFM2-350M
- **Метод**: LoRA (r=32, alpha=64)
- **Обучение**: Unsloth + curriculum learning (5 стадий по сложности)
- **Чекпоинт**: `checkpoints/lfm2-physics/final/`

#### 2. GPT From-scratch (не сошёлся)
- **Архитектура**: 6 слоёв, 6 голов, 384 dim (~11M параметров)
- **Токенизатор**: Digit-level (94 токена)
- **Проблема**: Выдаёт мусор/UNK токены
- **Чекпоинт**: `checkpoints/gpt-physics/best_model.pt`

## Evaluation Pipeline

### Компоненты
1. **MetricsComputer** — position/velocity MSE, energy/momentum conservation
2. **RolloutEvaluator** — авторегрессивные multi-step предсказания
3. **OOD Generator** — 8 типов out-of-distribution тестов
4. **Report Generator** — markdown отчёты с pandas таблицами
5. **Visualization** — GIF траекторий predicted vs ground truth

### Методология оценки

#### Single-step prediction
- Даём модели 3 кадра контекста (ground truth)
- Предсказываем следующий кадр
- Сравниваем с реальным следующим кадром
- **Без накопления ошибок**

#### Multi-step rollout
- Начинаем с ground truth контекста
- Каждое предсказание становится новым контекстом
- Ошибки накапливаются со временем
- Тестирует устойчивость модели

## Результаты

### LFM2 Fine-tuned Model

| Метрика | Значение |
|---------|----------|
| Mean Position Error | **22.64 px** |
| MSE (min) | 8.71 |
| MSE (max) | 7868.85 |
| MSE (mean) | 2478.52 |
| Objects Parsed | 37/37 (100%) |

**Интерпретация:**
- При размере сцены 800×600 px, ошибка ~23 px это ~3% по диагонали
- Модель корректно предсказывает формат вывода
- Большинство объектов предсказаны точно
- Есть outliers с большими ошибками (возможно столкновения)

### GPT From-scratch Model
- **Статус**: Не работает
- Выдаёт `<UNK>` токены и мусорные числа
- Требует дообучения или отладки training pipeline

## Визуализация

### Созданные GIF-ы

| Файл | Описание | Размер |
|------|----------|--------|
| `physics_eval_v2.gif` | Single-step, исправленный парсинг | 302 KB |
| `physics_single_step.gif` | Single-step, первая версия | 431 KB |
| `physics_comparison.gif` | Autoregressive rollout | 133 KB |

### Структура визуализации
- **Левая панель**: Ground Truth (зелёные точки)
- **Центральная панель**: LLM Prediction (синие точки)
- **Правая панель**: Overlay с красными линиями ошибок

## Технические детали

### Окружение
- **GPU**: NVIDIA RTX A6000 (48GB)
- **PyTorch**: 2.10.0+cu128
- **Transformers**: 4.57.6
- **Python**: 3.12

### Исправленные баги во время eval
1. `create_config` → `create_model_config` в run_evaluation.py
2. `callable(tokenizer)` check для различения HF vs PhysicsTokenizer
3. `vocab_size` inference из checkpoint для GPT модели

## Выводы

1. **LFM2 fine-tuning работает** — модель способна предсказывать физику с разумной точностью
2. **From-scratch требует работы** — текущий чекпоинт не функционален
3. **Evaluation pipeline готов** — все компоненты (metrics, rollout, OOD, viz) реализованы
4. **Основная сложность** — парсинг структурированного вывода LLM

## Следующие шаги

- [ ] Отладить GPT from-scratch training
- [ ] Провести полный OOD evaluation
- [ ] Тестировать на более длинных rollouts (100+ кадров)
- [ ] Добавить энергию/momentum metrics в визуализацию
- [ ] Сравнить несколько моделей в одном отчёте

---
*Отчёт сгенерирован: 2026-02-04*
*Проект: Physics LLM*
