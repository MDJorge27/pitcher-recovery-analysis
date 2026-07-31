import pandas as pd
import warnings
warnings.filterwarnings('ignore')

tj = pd.read_csv(r'C:\Users\micha\OneDrive\Programming\pitcher-recovery-analysis\tj_surgeries.csv', header=1)

mlb_tj = tj[(tj['Level'] == 'MLB') & (tj['Position'] == 'P')].copy()
mlb_tj['TJ Surgery Date'] = pd.to_datetime(mlb_tj['TJ Surgery Date'], errors='coerce')

mlb_tj = mlb_tj[
    (mlb_tj['TJ Surgery Date'].dt.year >= 2015) &
    (mlb_tj['TJ Surgery Date'].dt.year <= 2021)
].copy()

# Use Return Date as success indicator
mlb_tj['Return Date (same level)'] = pd.to_datetime(mlb_tj['Return Date (same level)'], errors='coerce')
mlb_tj['successful_return'] = mlb_tj['Return Date (same level)'].notna().astype(int)

# Keep only pitchers with valid mlbamid
mlb_tj = mlb_tj[mlb_tj['mlbamid'].notna()].copy()
mlb_tj['mlbamid'] = mlb_tj['mlbamid'].astype(int)

print(f"Total pitchers: {len(mlb_tj)}")
print(f"Successful returns: {mlb_tj['successful_return'].sum()}")
print(f"Did not return: {(mlb_tj['successful_return']==0).sum()}")
print(f"Success rate: {mlb_tj['successful_return'].mean():.1%}")
print(f"\nSample:")
print(mlb_tj[['Player', 'TJ Surgery Date', 'Return Date (same level)', 'successful_return', 'mlbamid']].head(10))