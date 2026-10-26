# 🏠 RentRight — Fair Rent Checker for India

People moving to a new city for college or a job often don't know what a *fair* rent is, and get overcharged.
RentRight predicts the **fair monthly rent** of a flat and tells you whether a listing is a **Bargain, Fair, or Overpriced**.

## What it does
| Model | Task | Output |
|---|---|---|
| Regression (Gradient Boosting) | Predict fair rent from flat details | Estimated rent in ₹ |
| Classification (Random Forest) | Judge a listing against its fair rent | Bargain / Fair / Overpriced |

## Data
~4,700 rental listings from 6 Indian cities (Kaggle: *House Rent Prediction Dataset*): BHK, size, floor, area type, city,
furnishing, tenant type, bathrooms, point of contact, rent.

## Method
1. Cleaned data, parsed the `Floor` text into numbers, removed extreme outliers (top/bottom 1% rent).
2. EDA to find what drives rent (city, size, bathrooms, furnishing).
3. Trained a baseline + Linear Regression + Random Forest + Gradient Boosting on `log(rent)`; 80/20 train-test split.
4. Built classification labels from **out-of-fold** regression predictions (rent ÷ fair rent: >1.25 Overpriced, <0.80 Bargain) to avoid data leakage.
5. Trained a Random Forest classifier and evaluated it against a majority-class baseline.

## Results (unseen test data)
| Regression | R² | MAE (₹) |
|---|---|---|
| Baseline (median) | -0.14 | 21,264 |
| Linear Regression | 0.73 | 9,694 |
| Random Forest | 0.77 | 9,314 |
| **Gradient Boosting** | **0.78** | **9,054** |

| Classification | Accuracy | Macro-F1 |
|---|---|---|
| Baseline (always "Fair") | 0.52 | 0.23 |
| **Random Forest** | **0.70** | **0.70** |

## Limitations (honest notes)
- Only ~4,700 listings from 2022; rents change over time.
- `Area Locality` was not used (too many unique values).
- Classification labels are derived from the regression model, not human-labelled.
- `Point of Contact` shows correlation with rent, not causation.

## Run it
Open `RentRight_Notebook.ipynb` in Google Colab → **Runtime → Run all**. The notebook downloads the dataset itself.

## Next steps
Locality features · hyper-parameter tuning · Flask API · React web app · deployment.

## Tech
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn
