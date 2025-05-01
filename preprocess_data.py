import pandas as pd
import numpy as np
import json

def preprocess_crop_data():
    # Load the dataset
    df = pd.read_csv('data/Crop_recommendation.csv')
    
    # Calculate statistics for each crop
    crop_stats = df.groupby('label').agg({
        'N': ['mean', 'std'],
        'P': ['mean', 'std'],
        'K': ['mean', 'std'],
        'temperature': ['mean', 'std'],
        'humidity': ['mean', 'std'],
        'ph': ['mean', 'std'],
        'rainfall': ['mean', 'std']
    }).round(2)
    
    # Create a dictionary of crop information
    crop_info = {}
    for crop in crop_stats.index:
        crop_info[crop] = {
            'N': {
                'mean': float(crop_stats.loc[crop, ('N', 'mean')]),
                'std': float(crop_stats.loc[crop, ('N', 'std')])
            },
            'P': {
                'mean': float(crop_stats.loc[crop, ('P', 'mean')]),
                'std': float(crop_stats.loc[crop, ('P', 'std')])
            },
            'K': {
                'mean': float(crop_stats.loc[crop, ('K', 'mean')]),
                'std': float(crop_stats.loc[crop, ('K', 'std')])
            },
            'temperature': {
                'mean': float(crop_stats.loc[crop, ('temperature', 'mean')]),
                'std': float(crop_stats.loc[crop, ('temperature', 'std')])
            },
            'humidity': {
                'mean': float(crop_stats.loc[crop, ('humidity', 'mean')]),
                'std': float(crop_stats.loc[crop, ('humidity', 'std')])
            },
            'ph': {
                'mean': float(crop_stats.loc[crop, ('ph', 'mean')]),
                'std': float(crop_stats.loc[crop, ('ph', 'std')])
            },
            'rainfall': {
                'mean': float(crop_stats.loc[crop, ('rainfall', 'mean')]),
                'std': float(crop_stats.loc[crop, ('rainfall', 'std')])
            }
        }
    
    # Save the processed data to a JSON file
    with open('data/processed_crop_data.json', 'w') as f:
        json.dump(crop_info, f, indent=4)
    
    print("Data preprocessing completed successfully!")
    print("\nSample of processed data:")
    print(json.dumps({list(crop_info.keys())[0]: crop_info[list(crop_info.keys())[0]]}, indent=4))

if __name__ == '__main__':
    preprocess_crop_data() 