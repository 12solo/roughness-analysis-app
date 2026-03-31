import pandas as pd
import numpy as np

class RoughnessLoader:
    def __init__(self):
        # Standardize naming
        self.param_map = {
            'ra': 'Ra', 'average roughness': 'Ra',
            'rq': 'Rq', 'rms': 'Rq',
            'rz': 'Rz', 'max height': 'Rz',
            'rt': 'Rt'
        }

    def process_files(self, uploaded_files, metadata_list):
        combined_data = []
        
        for file, meta in zip(uploaded_files, metadata_list):
            try:
                # Load Excel - assumes Params on Sheet 1, Profile on Sheet 2 (or configurable)
                df_params = pd.read_excel(file, sheet_name=0)
                
                # Fuzzy matching for columns
                row_data = meta.copy()
                for col in df_params.columns:
                    clean_col = str(col).lower().strip()
                    if clean_col in self.param_map:
                        row_data[self.param_map[clean_col]] = df_params[col].iloc[0]
                
                combined_data.append(row_data)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                
        return pd.DataFrame(combined_data)

    def extract_profile(self, file):
        """Extracts the continuous profile data (Length vs Amplitude)"""
        try:
            # Assumes Profile is in the second sheet or specific columns
            df = pd.read_excel(file, sheet_name=0) # Adjust index as needed
            # Look for numeric columns with many rows
            return df.select_dtypes(include=[np.number])
        except:
            return pd.DataFrame()