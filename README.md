# Rossmann Demand Forecasting

Forecasting daily store sales for the Rossmann drugstore chain, so the
business can plan staffing and stock ahead of time.

## Layout

- `src/data_loader.py` - reads, merges and cleans the raw files
- `src/features.py` - feature engineering with train-only fitted statistics
- `notebooks/01_eda.ipynb` - exploratory analysis and data validation

## How to run

1. Install dependencies: `pip install -r requirements.txt`
2. Download the data: `kaggle competitions download -c rossmann-store-sales`
3. Unzip `train.csv` and `store.csv` into `data/`.
4. Run the notebooks in `notebooks/` top to bottom.

The raw CSVs are not committed: the Kaggle competition rules restrict
redistributing them, so each user downloads their own copy.

## Notes

- `Customers` is deliberately unused: it is unknown at prediction time, so
  training on it would leak the answer.
- Closed days are dropped; they carry no sales signal and RMSPE divides by
  actual sales.
