# Wine Quality

Machine learning project on the UCI Wine Quality dataset (red and white Portuguese "Vinho Verde"
wines). We predict the quality score from 11 physicochemical measurements, both as a regression
problem and as a 3-class classification problem (Low / Medium / High), and we also look for groups
of wines with clustering and PCA.

## Structure

```
data/
  raw/              original UCI files (red, white and the dataset description)
  wine_clean.csv    combined dataset without duplicated rows (made in notebook 01)
notebooks/
  01_data_preparation.ipynb
  02_exploratory_data_analysis.ipynb
  03_regression_nested_cv.ipynb
  04_classification_nested_cv.ipynb
  05_clustering_pca.ipynb
  06_ablation_studies.ipynb
results/
  *.csv             tables with the results of each notebook
  figures/          plots saved by the notebooks
```

## How to run

```
pip install -r requirements.txt
cd notebooks
jupyter lab
```

Run the notebooks in order. Notebook 01 creates `data/wine_clean.csv`, and notebook 06 uses the
results saved by 03 and 04. All the models use `random_state=42`, so the results can be reproduced.

## Main results

Models are evaluated with nested cross-validation (5 outer folds, 3 inner folds with
`GridSearchCV`), on 5,320 wines after removing duplicates.

| Task | Best model | Score | Baseline |
|---|---|---|---|
| Regression | Random forest | RMSE 0.680, R² 0.40 | RMSE 0.880 |
| Classification | Random forest | Macro F1 0.604, accuracy 0.617 | Macro F1 0.203 |

- Alcohol is clearly the most important variable, followed by volatile acidity. The wine type
  (`is_red`) adds almost nothing.
- Keeping the duplicated rows makes the scores look better (RMSE 0.596, F1 0.718), but only because
  around 28% of the test wines have an identical copy in training.
- The clusters found with K-Means, Ward and GMM separate red and white wines, not quality.

## Data

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining
from physicochemical properties. Decision Support Systems, 47(4):547-553, 2009.
Available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality).
