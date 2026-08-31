# Rossmann Demand Forecasting

Rossmann is a large drugstore chain in Germany. This project predicts how much
each store will sell on each day.

The company needs this forecast to plan its work. When a store knows its future
sales, it can order the right amount of stock. It can also decide how many
people should work on each day.

## The problem

We predict daily sales for each store, six weeks into the future.

Six weeks is a long time, and this changes what we are allowed to use. For
example, we do not know how many customers will come to the store in six weeks.
So we cannot use that number, even though it is in the data. We only use
information that we already know today: the date, the store, and the promotions
that the company has planned.

## The data

The data comes from a Kaggle competition. It has about one million rows. Each
row is one store on one day.

| File | What it contains |
|---|---|
| `train.csv` | Daily sales of 1,115 stores, from January 2013 to July 2015 |
| `store.csv` | Facts about each store: its type, its product range, its competitors |
| `test.csv` | The days we must predict: August and September 2015 |

The CSV files are **not** in this repository. The Kaggle rules do not allow us
to share them. Please download them yourself. The steps are in the last section.

## How we measure the error

We use RMSPE (Root Mean Square Percentage Error).

This metric works with percentages, not with absolute numbers. An error of 500
units is small for a big store, but it is large for a small store. RMSPE looks
at both stores in a fair way.

This has one practical effect. We cannot divide by zero, so we cannot score days
with zero sales. Closed days are therefore removed from the data. For those days
we simply predict zero, and we do not need a model.

## What we found in the data

The notebook `notebooks/01_eda.ipynb` explores the data. These are the main
results.

**The data is clean.** All values are in a normal range. There are no rows that
contradict each other. Every store in the daily table also exists in the store
table. Only one value looks wrong: one store says that its competitor opened in
1900, which is clearly a placeholder.

**Sales follow a weekly pattern.** Monday is the strongest day, and sales go
down slowly until the weekend. Almost all stores close on Sunday. The few stores
that stay open are special, so we cannot compare their Sunday average with the
other days.

**December is the best month.** Sales rise a lot before Christmas and then fall
again in the quiet summer months. Over the two and a half years there is no
strong growth and no strong decline. Sales move because of the season and the
promotions, not because the business is growing.

**Promotions work well.** Stores sell clearly more on promotion days. This is a
useful feature, because the company plans its promotions in advance. We know
them at prediction time.

**Some stores are very different.** Store type `b` sells much more than the
others, but there are only 17 such stores. Product range `b` is even rarer, with
only 9 stores. These groups are too small for their own model, so we use them as
features instead.

**About 180 stores disappear for six months in 2014.** These stores were closed
for renovation. Their rows are simply missing from the data. This is important,
because those stores have a shorter history than the others.

**We must not use the `Customers` column.** The number of customers and the
sales of a store are almost the same information. If we train on it, our test
score looks excellent, but the model is useless. In real life we do not know the
number of customers six weeks in advance.

## Project structure

```
data/                 the raw CSV files (you download them, they are not shared here)
notebooks/
  01_eda.ipynb        data checks and exploration
  02_baseline.ipynb   the metric, the validation windows and the baseline
  03_xgboost.ipynb    the gradient boosting model and what it is worth
  04_segmentation.ipynb  grouping the stores, and one model per group
src/
  data_loader.py      reads the files, joins them, cleans them
  features.py         builds the features for the model
  metrics.py          RMSPE, and a way to split the error by group
  validation.py       cuts the data by time, in the shape of the real task
  model.py            XGBoost on the logarithm of the sales
  experiment.py       runs one approach on every fold and collects the scores
  segments.py         groups the stores by the way they react
requirements.txt      the Python packages you need
```

`src/data_loader.py` reads `train.csv` and `store.csv`. It fixes a small problem
in the raw file, where the column `StateHoliday` mixes the number `0` and the
text `"0"`. It then joins the store facts to every daily row, removes the closed
days, and sorts the rows by store and date.

`src/features.py` builds the features. It uses the `fit` and `transform` pattern
from scikit-learn:

- `fit` learns from the training data only. It calculates the average sales of
  each store, the average sales of each store on each weekday, and how much each
  store gains from a promotion.
- `transform` builds the simple features, such as the year, the month and the
  week, and then joins the values that `fit` has learned.

This split is not only a question of style. It protects us from leakage. Every
learned value comes from the training period, so nothing from the future can
enter the features.

`src/validation.py` cuts the data by date. The competition asks for six weeks
that we cannot see, so our validation windows are six weeks long too. We build
four of them, one after the other, going back in time. Each window trains on
every day before it starts. A random split would let the model see days that
come after the days it predicts, and the score would be far too good.

`src/metrics.py` holds RMSPE and a function that splits the error by any
column, for example by store type or by month. That breakdown tells us where
the model is weak.

`src/model.py` trains XGBoost on the logarithm of the sales, and turns the
prediction back at the end. It also chooses the number of trees on a small
inner window taken from the end of the training data, so that the validation
window stays untouched.

`src/experiment.py` runs one approach on all four windows and collects the
scores. It asks for a function that trains on the rows of a single fold, which
keeps the fitting inside the fold where it belongs.

`src/segments.py` puts every store into one of four groups. It describes a
store by ratios, such as how much a promotion helps it, so that the groups
follow the behaviour of a store and not its size.

## Results so far

Every model is measured on the four validation windows described above.

| Window ends | Baseline | XGBoost | XGBoost per segment |
|---|---|---|---|
| 2015-03-27 | 0.1877 | 0.1795 | 0.1759 |
| 2015-05-08 | 0.1746 | 0.1523 | 0.1441 |
| 2015-06-19 | 0.1584 | 0.1305 | 0.1260 |
| 2015-07-31 | 0.1449 | 0.1364 | 0.1345 |
| **Average** | **0.1664** | **0.1497** | **0.1451** |

A lower number is better. The baseline is simple: for each store, weekday and
promotion state, it predicts the median sales of the past.

The two XGBoost columns are the average of three random seeds. The model draws
80 percent of the columns for each tree, so a single run moves by a few
thousandths and one seed alone can flatter a result. Both models win on every
window, which was our rule for accepting a model.

Two decisions did the work.

**Training on the logarithm of the sales** is worth about 0.008. RMSPE reads
every error as a percentage, and the logarithm makes the training goal read
them the same way. Without it the model even loses to the baseline on the
hardest window.

**One model per store group** is worth another 0.005. Four models, one for each
behaviour group, beat a single model on every window. Handing the group to a
single model as a feature does almost nothing, so the grouping has to change
the shape of the model, not only the list of features.

The four windows are not equally hard, and the older ones have a larger error.
This is exactly why we use four windows and not one. For the baseline, a single
window would have reported 0.1449, which is the friendliest of the four
results.

The notebook `02_baseline.ipynb` tests three reasons for that difference. It is
not the length of the training history: when every window gets the same
training data, the trend stays. It is not the renovated stores either, because
they change the score by 0.0024 at most. What is left is the season. The spring
windows are simply harder to predict than the summer windows.

## Next steps

The metric, the validation setup, a working model and the store groups are
ready. The next steps are:

1. Look at the errors of the segmented model group by group, and let the weak
   spots choose the next features.
2. Add those features, for example the days to the next holiday and the length
   of a promotion period.
3. Tune the model parameters at the end, when the features are ready.
4. Correct the small bias that the logarithm leaves behind, first for all
   stores and then for each group.

Two results shape this list. We found a clear pattern in the raw data, built two
features for it, and measured no gain at all: the model knew it already through
the store averages. And we expected a single model to beat four smaller ones,
and it did not. Every idea now has to prove itself on the four windows, with
more than one seed, before it stays.

## How to run

1. Install the packages:
   ```
   pip install -r requirements.txt
   ```
2. Download the data from Kaggle:
   ```
   kaggle competitions download -c rossmann-store-sales
   ```
3. Unzip the files and put `train.csv`, `store.csv` and `test.csv` into the
   `data/` folder.
4. Open the notebooks in `notebooks/` and run them from top to bottom.
