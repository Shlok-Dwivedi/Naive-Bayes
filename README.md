# Lab 7: Gaussian Naive Bayes on the Raisin Dataset

Binary classification of raisin grains into **Kecimen** and **Besni** using a Gaussian Naive Bayes classifier from Scikit-learn.

## Overview

The notebook covers exploratory data analysis, label encoding, training a Gaussian Naive Bayes model, evaluating it, and checking how the `var_smoothing` hyperparameter affects performance.

**Dataset:** the Raisin Grains dataset (`Raisin_Dataset.xlsx - Raisin_Grains_Dataset.csv`) has 900 samples, 7 numerical features (Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, Extent, Perimeter) and a `Class` label. Each class has 450 samples. There are no missing values or duplicates.

## Workflow

1. Load the data, inspect it with `head()` and `tail()`, and check for missing values, duplicates and class counts
2. EDA: bar plot and pie chart of classes, histograms, boxplots, histograms with KDE by class, correlation heatmap, and scatter plots
3. Encode the `Class` column with `LabelEncoder` (Besni = 0, Kecimen = 1)
4. 80:20 stratified train-test split (720 train, 180 test)
5. Train `GaussianNB` and predict on the test set
6. Evaluate with accuracy, precision, recall, F1-score, a classification report and a confusion matrix
7. Compare training and testing accuracy
8. Try `var_smoothing` values from 1e-12 to 1e-2
9. Predict the class of a new sample entered by the user

## Results (from the notebook run)

Precision and recall are calculated for Kecimen (label 1).

| Metric | Value |
|--------|-------|
| Accuracy | 0.8167 |
| Precision | 0.7522 |
| Recall | 0.9444 |
| F1-score | 0.8374 |
| Training accuracy | 0.8264 |
| Testing accuracy | 0.8167 |

Confusion matrix: `[[62, 28], [5, 85]]`. The main source of error is Besni raisins predicted as Kecimen (28 cases).

**var_smoothing:** the smallest value tried, 1e-12, gave the best test accuracy (0.8333). The default 1e-9 gave 0.8167, and larger values lowered accuracy (0.7833 at 1e-2).

## Project structure

```
lab7-gaussian-nb-raisin/
├── Practical_7_ML_1oct_2026.ipynb                      # notebook
├── Raisin_Dataset.xlsx - Raisin_Grains_Dataset.csv     # dataset (add this yourself)
├── requirements.txt
└── README.md
```

## How to run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Practical_7_ML_1oct_2026.ipynb
```

Place the CSV file in the same folder as the notebook (or edit the path in the `read_csv` cell), then run all cells. The last cell uses `input()` and asks for the seven feature values of a new raisin. Use real-scale values (for example Area in the tens of thousands) to get a meaningful prediction.

## Notes

- The notebook output was generated in Google Colab, but it runs the same locally.
- The `var_smoothing` list in the notebook contains `1e-7` twice, so two rows of the results table are identical.
