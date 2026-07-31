import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pybaseball import statcast_pitcher, cache
cache.enable()

# Load TJ surgery database
tj = pd.read_csv(r'C:\Users\micha\OneDrive\Programming\pitcher-recovery-analysis\tj_surgeries.csv', header=1)

mlb_tj = tj[(tj['Level'] == 'MLB') & (tj['Position'] == 'P')].copy()
mlb_tj['TJ Surgery Date'] = pd.to_datetime(mlb_tj['TJ Surgery Date'], errors='coerce')

mlb_tj = mlb_tj[
    (mlb_tj['TJ Surgery Date'].dt.year >= 2015) &
    (mlb_tj['TJ Surgery Date'].dt.year <= 2021)
].copy()

mlb_tj['Return Date (same level)'] = pd.to_datetime(mlb_tj['Return Date (same level)'], errors='coerce')
mlb_tj['successful_return'] = mlb_tj['Return Date (same level)'].notna().astype(int)
mlb_tj = mlb_tj[mlb_tj['mlbamid'].notna()].copy()
mlb_tj['mlbamid'] = mlb_tj['mlbamid'].astype(int)

# Metrics to extract
METRICS = [
    'release_speed', 'release_spin_rate', 'release_extension',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z'
]

def get_pitcher_stats(player_id, start, end):
    try:
        data = statcast_pitcher(start, end, player_id)
        if data is None or len(data) == 0:
            return None
        
        # Filter to fastballs only
        fastballs = data[data['pitch_type'].isin(['FF', 'SI'])].copy()
        if len(fastballs) < 50:
            return None
        
        # Calculate zone percentage
        zone_pct = fastballs['zone'].notna().sum() / len(fastballs)
        in_zone = fastballs[fastballs['zone'].between(1, 9)]
        zone_pct = len(in_zone) / len(fastballs)
        
        # Summarize metrics
        stats = {}
        for metric in METRICS:
            if metric in fastballs.columns:
                stats[f'pre_{metric}'] = fastballs[metric].mean()
        
        stats['pre_zone_pct'] = zone_pct
        stats['pre_pitch_count'] = len(fastballs)
        return stats
    
    except Exception as e:
        return None

# Pull pre-surgery stats for all pitchers
results = []
total = len(mlb_tj)

for i, (_, row) in enumerate(mlb_tj.iterrows()):
    surgery_date = row['TJ Surgery Date']
    start = str((surgery_date - pd.DateOffset(years=1)).date())
    end = str(surgery_date.date())
    
    print(f"[{i+1}/{total}] {row['Player']}...")
    
    stats = get_pitcher_stats(int(row['mlbamid']), start, end)
    
    if stats:
        stats['Player'] = row['Player']
        stats['mlbamid'] = row['mlbamid']
        stats['surgery_date'] = surgery_date
        stats['surgery_year'] = surgery_date.year
        stats['age_at_surgery'] = row['Age']
        stats['successful_return'] = row['successful_return']
        results.append(stats)
        print(f"  Done - {stats['pre_pitch_count']} fastballs")
    else:
        print(f"  Skipped - insufficient data")

# Save results
df = pd.DataFrame(results)
df.to_csv('pre_surgery_stats.csv', index=False)

print(f"\nDone! Collected data for {len(df)} pitchers")
print(f"Successful returns: {df['successful_return'].sum()}")
print(f"Did not return: {(df['successful_return']==0).sum()}")
print(df.head())