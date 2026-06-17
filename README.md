# ЛР-3 SGD и его модификации

### Минимальный набор файлов:

- `src/` — весь код лабораторной (классы, функции, утилиты).  
- Отчёт в корне репозитория:
  - для **Python** — `report.ipynb` (шаблон уже есть);  
  - для **другого языка** — `report.pdf` / `report.md`.

### Рекомендуемая структура (Python):

```
.
├─ src/                 # код
├─ report.ipynb         # эксперименты (отчёт)
├─ requirements.txt     # зависимости
└─ README.md            # этот файл
```

### Пример настройки проекта на Python

Создайте виртуальное окружение:

```bash
python -m venv .venv
source .venv/bin/activate
```

Поставьте зависимости:

```bash
pip install -r requirements.txt
```

Запустите JupyterLab:

```bash
jupyter lab
```
