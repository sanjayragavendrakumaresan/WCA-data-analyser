"""Shared configuration for the WCA analysis package."""

from pathlib import Path

DEFAULT_DATA_PATH = Path(".")

FILE_CONFIGS = {
    "persons": {"file": "WCA_export_persons.tsv", "sep": "\t", "encoding": "utf-8"},
    "results": {"file": "WCA_export_results.tsv", "sep": "\t", "encoding": "utf-8"},
    "ranks_average": {"file": "WCA_export_ranks_average.tsv", "sep": "\t", "encoding": "utf-8"},
    "events": {"file": "WCA_export_events.tsv", "sep": "\t", "encoding": "utf-8"},
    "countries": {"file": "WCA_export_countries.tsv", "sep": "\t", "encoding": "utf-8"},
    "continents": {"file": "WCA_export_continents.tsv", "sep": "\t", "encoding": "utf-8"},
    "round_types": {"file": "WCA_export_round_types.tsv", "sep": "\t", "encoding": "utf-8"},
    "scrambles": {"file": "WCA_export_scrambles.tsv", "sep": "\t", "encoding": "utf-8"},
    "result_attempts": {"file": "WCA_export_result_attempts.tsv", "sep": "\t", "encoding": "utf-8"},
    "competitions": {"file": "WCA_export_competitions.tsv", "sep": "\t", "encoding": "utf-8"},
    "championships": {"file": "WCA_export_championships.tsv", "sep": "\t", "encoding": "utf-8"},
    "formats": {"file": "WCA_export_formats.tsv", "sep": "\t", "encoding": "utf-8"},
}
