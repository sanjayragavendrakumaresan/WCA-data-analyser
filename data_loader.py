"""Loading and preprocessing of WCA export files."""

import numpy as np
import pandas as pd
from pathlib import Path

from .config import FILE_CONFIGS


class WCADataLoader:
    def __init__(self, data_path=None):
        self.data_path = self._find_data_path(data_path)
        self.data = {}

    def _find_data_path(self, data_path):
        if data_path is not None:
            candidate = Path(data_path).expanduser().resolve()
            if candidate.is_dir() and any((candidate / config['file']).exists() for config in FILE_CONFIGS.values()):
                return candidate

        roots = [
            Path.cwd(),
            Path(__file__).resolve().parent,
            Path(__file__).resolve().parent / 'data',
        ]

        for root in roots:
            if root.is_dir() and any((root / config['file']).exists() for config in FILE_CONFIGS.values()):
                return root

        search_roots = [Path.cwd(), Path(__file__).resolve().parent]
        for root in search_roots:
            if not root.exists():
                continue
            for result_file in root.rglob('WCA_export_results.tsv'):
                return result_file.parent

        return Path.cwd()


    def load_all_files(self):
            """Load all TSV files"""
            print(f"\nLoading WCA data files from: {self.data_path}")

            file_configs = FILE_CONFIGS
            
            for key, config in file_configs.items():
                try:
                    df = pd.read_csv(
                        str(self.data_path / config['file']),
                        sep=config['sep'],
                        encoding=config['encoding'],
                        low_memory=False,
                        on_bad_lines='skip'
                    )
                    self.data[key] = df
                    print(f"   Loaded {key}: {len(df):,} records")
                except Exception as e:
                    print(f"   Error loading {key}: {str(e)}")
                    self.data[key] = pd.DataFrame()

            return self.data


    def preprocess_data(self):
            """Clean and preprocess the loaded data"""
            print("\nPreprocessing data...")

            if 'results' in self.data and not self.data['results'].empty:
                                                                  
                for col in ['best', 'average']:
                    if col in self.data['results'].columns:
                        self.data['results'][f'{col}_seconds'] = self.data['results'][col] / 100
                                              
                        mask = self.data['results'][col] > 99999999
                        self.data['results'].loc[mask, f'{col}_seconds'] = np.nan
                        self.data['results'].loc[self.data['results'][f'{col}_seconds'] < 0, f'{col}_seconds'] = np.nan

                                                   
                if 'competition_id' in self.data['results'].columns:
                                                               
                    self.data['results']['year'] = self.data['results']['competition_id'].str.extract(r'(\d{4})').astype(float)
                    self.data['results'].loc[self.data['results']['year'] < 2000, 'year'] = np.nan

                                       
            if 'events' in self.data and not self.data['events'].empty:
                self.event_names = dict(zip(self.data['events']['id'], self.data['events']['name']))
            else:
                self.event_names = {}

            print("Preprocessing complete.")
            return self.data
