# Certificate Project Report

**Author:** Omar Eid  
**Date:** 2025-09-13

## 1. Problem
- Music: predict mood (happy/sad/energetic/calm) from audio features.
- YouTube: classify comments into positive/negative/neutral.

## 2. Data
- Music: 200 synthetic rows (can be replaced by Spotify features).
- YouTube: 600 labeled synthetic comments (can be replaced by scraped/real comments).
- Files: `data/music/songs.csv`, `data/youtube/comments.csv`.

## 3. Cleaning & Preprocessing
- Deduplicate, handle NA/invalid types.
- For music, numeric features filled with median.
- For text, TF-IDF with bigrams.

## 4. Modeling
- Music: RandomForest with GridSearch (F1-macro).
- YouTube: LinearSVC with GridSearch (F1-macro).

## 5. Evaluation
- Metrics: accuracy, macro-F1, classification report, confusion matrix.
- See notebooks under `notebooks/` for outputs.

## 6. Results & Insights
- Include screenshots / copy metrics from your runs.
- Discuss feature importance (music) and top tokens (optional in text).

## 7. Conclusion & Future Work
- Collect more real data.
- Try balancing, error analysis, SHAP, or calibration.
