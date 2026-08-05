import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from pybaseball import statcast_pitcher, cache
import warnings

warnings.filterwarnings('ignore')
cache.enable()

# ==============================================================
# SECTION 1: LOAD AND PREPARE DATA
# ==============================================================

# Load TJ surgery database
tj = pd.read_csv('tj_surgeries.csv', header=1)
mlb_tj = tj[(tj['Level'] == 'MLB') & (tj['Position'] == 'P')].copy()
mlb_tj['TJ Surgery Date'] = pd.to_datetime(mlb_tj['TJ Surgery Date'], errors='coerce')
mlb_tj = mlb_tj[
    (mlb_tj['TJ Surgery Date'].dt.year >= 2015) &
    (mlb_tj['TJ Surgery Date'].dt.year <= 2021)
].copy()
mlb_tj['Return Date (same level)'] = pd.to_datetime(
    mlb_tj['Return Date (same level)'], errors='coerce'
)
mlb_tj['successful_return'] = mlb_tj['Return Date (same level)'].notna().astype(int)
mlb_tj = mlb_tj[mlb_tj['mlbamid'].notna()].copy()
mlb_tj['mlbamid'] = mlb_tj['mlbamid'].astype(int)

# ==============================================================
# SECTION 2: PITCHER STATS FUNCTION
# ==============================================================

def get_pitcher_stats(player_id, start, end, label='pre'):
    """
    Pulls all pitch type Statcast data for a given pitcher and date range.
    Returns aggregated metrics or None if insufficient data.
    """
    try:
        data = statcast_pitcher(start, end, int(player_id))

        if data is None or len(data) == 0:
            return None

        data = data[data['pitch_type'].notna()].copy()

        if len(data) < 50:
            return None

        fastballs = data[data['pitch_type'].isin(['FF', 'SI'])]

        if len(fastballs) < 30:
            return None

        in_zone = data[data['zone'].between(1, 9)]
        zone_pct = len(in_zone) / len(data)

        fb_in_zone = fastballs[fastballs['zone'].between(1, 9)]
        fb_zone_pct = len(fb_in_zone) / len(fastballs) if len(fastballs) > 0 else np.nan

        fastball_pct = len(fastballs) / len(data)
        pitch_type_count = data['pitch_type'].nunique()

        stats = {
            f'{label}_release_speed': fastballs['release_speed'].mean(),
            f'{label}_release_spin_rate': fastballs['release_spin_rate'].mean(),
            f'{label}_release_extension': fastballs['release_extension'].mean(),
            f'{label}_zone_pct': zone_pct,
            f'{label}_fb_zone_pct': fb_zone_pct,
            f'{label}_fastball_pct': fastball_pct,
            f'{label}_pitch_type_count': pitch_type_count,
            f'{label}_pitch_count': len(data)
        }

        return stats

    except Exception:
        return None

# ==============================================================
# SECTION 3: PULL PRE-SURGERY STATCAST DATA
# ==============================================================

pre_results = []
total = len(mlb_tj)

for i, (_, row) in enumerate(mlb_tj.iterrows()):
    surgery_date = row['TJ Surgery Date']
    start = str((surgery_date - pd.DateOffset(years=1)).date())
    end = str(surgery_date.date())

    print(f"[{i+1}/{total}] {row['Player']}...")

    stats = get_pitcher_stats(int(row['mlbamid']), start, end, label='pre')

    if stats:
        stats['Player'] = row['Player']
        stats['mlbamid'] = row['mlbamid']
        stats['surgery_date'] = surgery_date
        stats['surgery_year'] = surgery_date.year
        stats['age_at_surgery'] = row['Age']
        stats['successful_return'] = row['successful_return']
        stats['return_date'] = row['Return Date (same level)']
        pre_results.append(stats)
        print(f"  Done - {stats['pre_pitch_count']} pitches")
    else:
        print(f"  Skipped - insufficient data")

pre_df = pd.DataFrame(pre_results)
pre_df.to_csv('pre_surgery_stats.csv', index=False)

print(f"\nPre-surgery data collected for {len(pre_df)} pitchers")
print(f"Successful returns: {pre_df['successful_return'].sum()}")
print(f"Did not return: {(pre_df['successful_return']==0).sum()}")

# ==============================================================
# SECTION 4: PULL POST-SURGERY STATCAST DATA
# ==============================================================

returned = pre_df[pre_df['successful_return'] == 1].copy()
returned['return_date'] = pd.to_datetime(returned['return_date'], errors='coerce')

post_results = []
total_returned = len(returned)

for i, (_, row) in enumerate(returned.iterrows()):
    return_date = row['return_date']
    if pd.isna(return_date):
        continue

    start = str(return_date.date())
    end = str((return_date + pd.DateOffset(years=1)).date())

    print(f"[{i+1}/{total_returned}] {row['Player']} (post)...")

    stats = get_pitcher_stats(int(row['mlbamid']), start, end, label='post')

    if stats:
        stats['mlbamid'] = row['mlbamid']
        post_results.append(stats)
        print(f"  Done - {stats['post_pitch_count']} pitches")
    else:
        print(f"  Skipped - insufficient data")

post_df = pd.DataFrame(post_results)
df = pre_df.merge(post_df, on='mlbamid', how='left')

# ==============================================================
# SECTION 5: CALCULATE METRIC CHANGES
# ==============================================================

df['velocity_change'] = df['post_release_speed'] - df['pre_release_speed']
df['spin_rate_change'] = df['post_release_spin_rate'] - df['pre_release_spin_rate']
df['zone_pct_change'] = df['post_zone_pct'] - df['pre_zone_pct']
df['fb_zone_pct_change'] = df['post_fb_zone_pct'] - df['pre_fb_zone_pct']
df['fastball_pct_change'] = df['post_fastball_pct'] - df['pre_fastball_pct']
df['extension_change'] = df['post_release_extension'] - df['pre_release_extension']
df['pitch_type_count_change'] = df['post_pitch_type_count'] - df['pre_pitch_type_count']

df.to_csv('full_pitcher_stats.csv', index=False)

print(f"\nFull dataset saved with {len(df)} pitchers")
print(f"\nAverage metric changes for returned pitchers:")
returned_with_post = df[df['post_release_speed'].notna()]
print(f"  Velocity change: {returned_with_post['velocity_change'].mean():.2f} mph")
print(f"  Zone pct change: {returned_with_post['zone_pct_change'].mean():.3f}")
print(f"  Fastball pct change: {returned_with_post['fastball_pct_change'].mean():.3f}")

# ==============================================================
# SECTION 6: PRE-SURGERY MODEL
# ==============================================================

pre_features = [
    'pre_release_speed',
    'pre_release_spin_rate',
    'pre_release_extension',
    'pre_zone_pct',
    'pre_fb_zone_pct',
    'pre_fastball_pct',
    'pre_pitch_type_count',
    'age_at_surgery',
    'surgery_year'
]

model_df = df.dropna(subset=pre_features + ['successful_return']).copy()

print(f"\nPitchers in pre-surgery model: {len(model_df)}")
print(f"Successful returns: {model_df['successful_return'].sum()}")
print(f"Did not return: {(model_df['successful_return']==0).sum()}")

X_pre = model_df[pre_features].copy()
y_pre = model_df['successful_return'].copy()

imputer_pre = SimpleImputer(strategy='median')
X_pre_imputed = imputer_pre.fit_transform(X_pre)

X_train, X_test, y_train, y_test = train_test_split(
    X_pre_imputed, y_pre, test_size=0.2, random_state=42, stratify=y_pre
)

pre_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
pre_model.fit(X_train, y_train)

y_pred = pre_model.predict(X_test)
cv_scores = cross_val_score(pre_model, X_pre_imputed, y_pre, cv=5)

print("\n=== Pre-Surgery Model Results ===")
print(f"\nCross-validation accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
print(f"\nTest set results:")
print(classification_report(y_test, y_pred, target_names=['Did Not Return', 'Returned']))

pre_importance = pd.DataFrame({
    'feature': pre_features,
    'importance': pre_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nPre-Surgery Feature Importance:")
print(pre_importance)

plt.figure(figsize=(12, 7))
plt.barh(pre_importance['feature'], pre_importance['importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Pre-Surgery Predictors of Successful TJ Recovery')
plt.tight_layout()
plt.savefig('pre_surgery_feature_importance.png', dpi=150)
plt.show()

# ==============================================================
# SECTION 7: POST-SURGERY MODEL
# ==============================================================

# Post-surgery model uses change metrics to predict long-term success
# Defines long-term success as returning AND maintaining performance
# Uses pitchers who returned and have post-surgery data

post_features = [
    'pre_release_speed',
    'pre_release_spin_rate',
    'pre_fb_zone_pct',
    'age_at_surgery',
    'velocity_change',
    'spin_rate_change',
    'fb_zone_pct_change',
    'zone_pct_change',
    'fastball_pct_change',
    'extension_change',
    'pitch_type_count_change'
]

# For the post-surgery model, define long-term success more rigorously
# A pitcher who returned but threw fewer than 100 pitches post-surgery
# is not truly a successful comeback
post_model_df = df[df['post_release_speed'].notna()].copy()

# Define long-term success: returned AND threw 200+ post-surgery pitches
# This filters out pitchers who returned briefly then were cut
post_model_df['long_term_success'] = (
    (post_model_df['successful_return'] == 1) &
    (post_model_df['post_pitch_count'] >= 200)
).astype(int)

post_model_df = post_model_df.dropna(subset=post_features).copy()

print(f"\nPitchers in post-surgery model: {len(post_model_df)}")
print(f"Long-term success: {post_model_df['long_term_success'].sum()}")
print(f"Did not achieve long-term success: {(post_model_df['long_term_success']==0).sum()}")

X_post = post_model_df[post_features].copy()
y_post = post_model_df['long_term_success'].copy()

imputer_post = SimpleImputer(strategy='median')
X_post_imputed = imputer_post.fit_transform(X_post)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_post_imputed, y_post, test_size=0.2, random_state=42, stratify=y_post
)

post_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
post_model.fit(X_train_p, y_train_p)

y_pred_p = post_model.predict(X_test_p)
cv_scores_p = cross_val_score(post_model, X_post_imputed, y_post, cv=5)

print("\n=== Post-Surgery Model Results ===")
print(f"\nCross-validation accuracy: {cv_scores_p.mean():.1%} (+/- {cv_scores_p.std():.1%})")
print(f"\nTest set results:")
print(classification_report(y_test_p, y_pred_p, 
      target_names=['Did Not Sustain Return', 'Long-Term Success']))

post_importance = pd.DataFrame({
    'feature': post_features,
    'importance': post_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nPost-Surgery Feature Importance:")
print(post_importance)

plt.figure(figsize=(12, 7))
plt.barh(post_importance['feature'], post_importance['importance'], color='darkgreen')
plt.xlabel('Importance')
plt.title('Post-Surgery Predictors of Long-Term MLB Success After TJ')
plt.tight_layout()
plt.savefig('post_surgery_feature_importance.png', dpi=150)
plt.show()

# ==============================================================
# SECTION 8: VISUALIZE PRE VS POST METRIC CHANGES
# ==============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('How Pitcher Metrics Change After Tommy John Surgery', fontsize=14)

change_metrics = [
    ('velocity_change', 'Velocity Change (mph)'),
    ('fb_zone_pct_change', 'Fastball Zone % Change'),
    ('spin_rate_change', 'Spin Rate Change (rpm)'),
    ('fastball_pct_change', 'Fastball Usage Change')
]

for ax, (metric, label) in zip(axes.flatten(), change_metrics):
    data = returned_with_post[metric].dropna()
    ax.hist(data, bins=20, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax.axvline(x=data.mean(), color='green', linestyle='-', 
               linewidth=2, label=f'Mean: {data.mean():.3f}')
    ax.set_xlabel(label)
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_title(label)

plt.tight_layout()
plt.savefig('metric_changes.png', dpi=150)
plt.show()

print("\nAll charts saved!")
print("\n=== Project Summary ===")
print(f"Pre-surgery model accuracy: {cv_scores.mean():.1%}")
print(f"Post-surgery model accuracy: {cv_scores_p.mean():.1%}")
print(f"Total pitchers analyzed: {len(df)}")
print(f"Successful returns: {df['successful_return'].sum()}")
print(f"Success rate: {df['successful_return'].mean():.1%}")