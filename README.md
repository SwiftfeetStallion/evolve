Установка:

```
git clone https://github.com/SwiftfeetStallion/evolve.git
cd evolve

python -m venv env
env/Scripts/activate
pip install -r requirements.txt
```

Структура папок:

```
evolve/
├── algorithms
├── code
├── data
└── evolve_config/
    └── config.yml
└── results/
    └── algorithms/
    └── datasets/
    └── HGS/
└── scripts
└── validators/
    └── evaluator.py 
├── test.ipynb  
```
Содержимое: 

* В папке `algorithms` есть реализации некоторых алгоритмов.
* В папке `data` собраны данные с задачами.
* В папке `results` сохранены статистики по алгоритмам и отдельно по наборам данных.
* В папке `validators` находится валидатор для эволюции (можно написать и другой).
* В папке `code` представлены некоторые начальные программы для эволюции.
* Папка `evolve_config` содержит настройки для эволюции.
* Файл `test.ipynb` - пример работы полученного в ходе эволюции решения.

Код для запуска алгоритмов Clarke-Wright (CW), Hybrid genetic search (HGS), OR-tools (OR) и эволюции (evolve) находится в папке `scripts` в файлах с соответствующими именами. Можно менять глобальные переменные.

Пример запуска: `python scripts/run_evolve.py`.

Для эволюции необходимо настроить `config.yml`, а в файле `run_evolve.py` указать путь к валидатору и начальному решению, количество итераций и директорию для логирования. При использовании валидатора из `evaluator.py` нужно уточнить пути к задачам, которые требуется оптимизировать, добавив их в массив `paths`.


