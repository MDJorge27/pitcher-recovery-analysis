# Pitcher Recovery Analysis

## Overview
This project predicts whether a MLB pitcher will successfully return 
after Tommy John surgery using Statcast performance metrics.

## Motivation
During my internship with a baseball agency, I collected Trackman data 
for a pitcher rehabbing from Tommy John surgery. His velocity and spin 
rate were encouraging, but he never got a chance to return to the MLB.

That experience raised a question I couldn't answer with the data I had: 
what actually predicts a successful comeback? This project is my attempt 
to find out.

## Data Sources
- **Tommy John Surgery Database** — Jon Roegele's publicly maintained 
  spreadsheet of MLB pitchers who underwent TJ surgery, including surgery 
  dates and return dates
- **MLB Statcast** — pitch-by-pitch data pulled via pybaseball, including 
  velocity, spin rate, extension, movement, and command metrics

## Methodology
1. Identified 162 MLB pitchers who underwent TJ surgery between 2015-2021
2. Pulled pre-surgery Statcast data for each pitcher (1 year prior to surgery)
3. Defined successful return as having a confirmed MLB return date
4. Engineered features: velocity, spin rate, extension, zone percentage, age
5. Trained a Random Forest classifier to predict successful return

## Key Findings
- **Zone percentage** (command) is the strongest predictor of successful 
  return — stronger than velocity or spin rate
- **Age at surgery** is the weakest predictor among the features tested, 
  suggesting teams may be overweighting age in roster decisions
- Model accuracy: 70.8% cross-validated

## Requirements
```bash
pip install pybaseball pandas matplotlib scikit-learn seaborn
```

## How to Run
```bash
python pitcher_recovery.py
```

## Author
Michael Jorge | [GitHub](https://github.com/MDJorge27)
