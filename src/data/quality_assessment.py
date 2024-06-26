from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


class QualityAssessment:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.save_path = Path(config['path']['eval_report'])

    def generate_exploratory_report(self, df, report_name, info_name):
        """Generate exploratory report"""
        self.logger.info(
            f'Generating exploratory report. See results in ``{self.save_path}`` directory')
        report_data = {
            'Data_Type': [],
            'Zero_Count': [],
            'Zero_Percentage': [],
            'NaN_Count': [],
            'NaN_Percentage': [],
            'Duplicate_Count': [],
            'Duplicate_Percentage': [],
            'Unique_Count': [],
            'Unique_Percentage': []
        }

        # Eval Metric Calcs
        for col in df.columns:
            total_rows = len(df)

            col_dtype = df[col].dtype

            zero_count = (df[col] == 0).sum()
            zero_percent = round((zero_count/total_rows)*100, 2)

            nan_count = df[col].isna().sum()
            nan_percent = round((nan_count/total_rows)*100, 2)

            duplicate_count = df[col].duplicated(keep='first').sum()
            duplicate_percent = round((duplicate_count/total_rows)*100, 2)

            unique_count = df[col].nunique()
            unique_percent = round((unique_count/total_rows)*100, 2)

            metrics = [
                col_dtype, zero_count, zero_percent,
                nan_count, nan_percent, duplicate_count,
                duplicate_percent, unique_count, unique_percent]

            for key, metric in zip(report_data.keys(), metrics):
                report_data[key].append(metric)

        report_df = pd.DataFrame(report_data, index=df.columns)
        statistics_df = round(df.describe(), 2)

        report_path = Path(f'{self.save_path}/{report_name}')
        statistics_path = Path(f'{self.save_path}/{info_name}')

        if os.path.isfile(report_path):
            pass
        else:
            report_df.to_excel(report_path, index=True)
        if os.path.isfile(statistics_path):
            pass
        else:
            statistics_df.to_excel(statistics_path, index=True)
