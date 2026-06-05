"""Inference-only survival dataset.

Skips training-specific columns (`discrete_label`, label-mapping) so test CSVs
only need `case_id`, the event-time column, and `censored`.
"""

from pathlib import Path

import torch

from hipt.src.data.dataset import DatasetOptions


class InferenceSurvivalDataset(torch.utils.data.Dataset):
    def __init__(self, options: DatasetOptions):
        self.options = options
        self.df = self._filter_missing_features(options.df.copy())

    def _filter_missing_features(self, df):
        present = [
            cid
            for cid in df.case_id
            if Path(self.options.features_dir, f"{cid}.pt").is_file()
        ]
        out = df[df.case_id.isin(present)].reset_index(drop=True)
        if len(out) != len(df):
            print(
                f"WARNING: {len(df) - len(out)} slides dropped because .pt files missing"
            )
        return out

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        feature = torch.load(
            Path(self.options.features_dir, f"{row.case_id}.pt"), map_location="cpu"
        )
        event_time = row[self.options.label_name]
        censored = row.censored
        # placeholder label slot — inference loops discard it
        return idx, feature, 0, event_time, censored
