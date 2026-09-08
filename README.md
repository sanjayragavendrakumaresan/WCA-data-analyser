# WCA Data Analysis and Prediction Suite

A menu-driven Python application for exploring **World Cube Association (WCA)** competition data. It provides competitor analysis, country and event statistics, performance plots, improvement trends, podium estimates, world-record forecasting, and competitor clustering.

The original single-file script has been reorganized into focused modules. The application behavior and interactive menu flow remain the same.

## 1. Requirements

The project requires:

- Python 3.10 or newer
- The WCA Results Export TSV files
- The Python packages listed in `requirements.txt`

The project uses pandas and NumPy for data processing, Matplotlib and Seaborn for charts, SciPy for statistics, and scikit-learn for regression, clustering, scaling, and dimensionality reduction.

## 2. Download the WCA dataset

The dataset comes from the official [World Cube Association Results Export page][1]. The WCA provides the export in SQL and TSV formats. This project uses the **TSV export** because the loader reads tab-separated files with pandas.

Download the latest TSV archive from the WCA export page, or use the current TSV permalink:

- [WCA Results Export page][1]
- [Latest WCA TSV export][2]

The archive contains files similar to the following:

```text
WCA_export_persons.tsv
WCA_export_results.tsv
WCA_export_ranks_average.tsv
WCA_export_events.tsv
WCA_export_countries.tsv
WCA_export_continents.tsv
WCA_export_round_types.tsv
WCA_export_scrambles.tsv
WCA_export_result_attempts.tsv
WCA_export_competitions.tsv
WCA_export_championships.tsv
WCA_export_formats.tsv
```

Extract the archive before running the application. The filenames above must remain unchanged because they are configured in `wca_analysis/config.py`.

> **Dataset note:** The WCA export is updated periodically. The export page includes the export date and format version. If the WCA changes the major export format version, review the project’s loader and preprocessing assumptions before using the new data.

## 3. Recommended project layout

Place the extracted TSV files in a separate data directory:

```text
project-root/
├── data/
│   ├── WCA_export_persons.tsv
│   ├── WCA_export_results.tsv
│   ├── WCA_export_events.tsv
│   └── ...
└── wca_analysis/
    ├── __init__.py
    ├── app.py
    ├── competitor_analyzer.py
    ├── config.py
    ├── data_loader.py
    ├── main.py
    ├── menu_base.py
    ├── predictive_menu.py
    └── statistics_menu.py
```

The default application entry point currently looks for the data files in the current working directory. Therefore, the simplest arrangement is to place the TSV files in the directory from which you launch the program.

For a separate `data/` directory, update the loader initialization in `wca_analysis/menu_base.py`:

```python
self.loader = WCADataLoader(data_path="./data")
```

## 4. Installation

Open a terminal in the directory containing the `wca_analysis` package and install the dependencies:

```bash
python -m pip install -r wca_analysis/requirements.txt
```

On some systems, use `python3` and `pip3` instead:

```bash
python3 -m pip install -r wca_analysis/requirements.txt
```

Using a virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell
python -m pip install -r wca_analysis/requirements.txt
```

## 5. Run the application

From the parent directory of `wca_analysis`, run:

```bash
python -m wca_analysis.main
```

If you are already inside the `wca_analysis` directory on Windows PowerShell or another terminal, use the included launcher instead:

```bash
python run.py
```

The launcher adds the project’s parent directory to Python’s import path automatically, so you do not need to move to the parent directory first.

The application loads the TSV files, preprocesses result times, and displays the main menu:

```text
1. Analyze Competitor by WCA ID
2. Global Statistics & Trends
3. Country Performance Analysis
4. Event Performance Analysis
5. Predictive Models
6. Exit
```

Follow the prompts shown in the terminal. Matplotlib charts will open in separate windows when an analysis includes visualizations.

## 6. Run from Python code

The application can also be imported and started programmatically:

```python
from wca_analysis import WCAMenu

app = WCAMenu()
app.run()
```

For direct data access:

```python
from wca_analysis.data_loader import WCADataLoader
from wca_analysis.competitor_analyzer import CompetitorAnalyzer

loader = WCADataLoader(data_path="./data")
data = loader.load_all_files()
data = loader.preprocess_data()

analyzer = CompetitorAnalyzer(data)
results = analyzer.get_competitor_results("2003BELL01")
print(results.head())
```

## 7. Module overview

| Module | Responsibility |
|---|---|
| `config.py` | Centralizes expected WCA filenames and parsing settings. |
| `data_loader.py` | Loads TSV files and preprocesses centisecond times and competition years. |
| `competitor_analyzer.py` | Provides competitor lookup, result filtering, trend analysis, predictions, and competitor-level plots. |
| `menu_base.py` | Defines the shared application state, initialization, main menu, and competitor menu. |
| `statistics_menu.py` | Contains global, country, and event statistics menus. |
| `predictive_menu.py` | Contains predictive-model, world-record forecast, and clustering menus. |
| `app.py` | Combines the menu base class and menu mixins into `WCAMenu`. |
| `main.py` | Provides the `python -m wca_analysis.main` entry point. |
| `run.py` | Provides the `python run.py` entry point when launched from inside the project directory. |
| `requirements.txt` | Lists the required third-party Python packages. |

## 8. Preprocessing performed by the application

The WCA export stores most timed results as integer centiseconds. The loader creates `best_seconds` and `average_seconds` columns by dividing valid values by 100. It also converts competition IDs into a `year` field where possible.

The application treats invalid or unavailable times as missing values. The original script’s filtering rules exclude negative values and unusually large encoded values from ordinary time-based analysis.

## 9. Troubleshooting

### `FileNotFoundError` or empty datasets

Confirm that the TSV files have been extracted and that their names match the names in `wca_analysis/config.py`. Also confirm that the terminal’s current working directory is the directory containing those files, or configure `WCADataLoader(data_path="/path/to/data")`.

### `ModuleNotFoundError`

Install the dependencies again from the project root:

```bash
python -m pip install -r wca_analysis/requirements.txt
```

### Charts do not appear

Run the application in an environment with a graphical display. On headless servers, configure a non-interactive Matplotlib backend and save figures to files instead of calling `plt.show()`.

### The application is slow during initial loading

The WCA export contains a large historical dataset. Loading all TSV files can require substantial memory and may take time, especially on the first run.

## 10. Data attribution and terms of use

This project uses public information from the World Cube Association database export. Follow the WCA’s terms of use and attribution requirements when redistributing analysis or derived results. The WCA export page contains the authoritative description, version information, acknowledgements, and usage conditions.

## References

[1]: https://www.worldcubeassociation.org/export/results "World Cube Association Results Export"
[2]: https://exports.worldcubeassociation.org/results/WCA_export_v2_251_20260908T040114Z.tsv.zip "World Cube Association TSV Results Export, September 2026"
