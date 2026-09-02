# Rossmann Demand Forecasting

Rossmann is a large drugstore chain in Germany. This project predicts how much
each of its 1,115 stores will sell on each day, six weeks into the future.

The company needs this forecast to plan its work. When a store knows its future
sales, it can order the right amount of stock. It can also decide how many
people should work on each day.

## In short

- **The final model scores 0.1189 RMSPE** on four time-ordered validation
  windows. A strong baseline scores 0.1664, so the model is 29 percent better.
- **The error analysis did the work.** The largest single gain, 0.024, came
  from six small columns about closed days. We built them because a notebook
  showed where the model was failing, not because we guessed.
- **Every change had to prove itself** on four windows, with three random
  seeds, and win on every window. Ideas that did not pass were kept in the
  notebooks with their numbers, so the failures are part of the record.
- **Tuning and calibration came last, and added nothing.** The settings we
  started with won the search, and the standard correction for the logarithm
  made the score worse. Every gain came from the data and the features.
- **No leakage.** The model uses only what the company knows six weeks ahead:
  the date, the store, the planned promotions and the planned opening days.
- **The model ships.** The final model is trained once, saved with a model
  card, and used from a file to forecast the test period and to score single
  new rows. A check shows that the saved file gives the same score as the
  notebooks.

## The problem

We predict daily sales for each store, six weeks into the future.

Six weeks is a long time, and this changes what we are allowed to use. For
example, we do not know how many customers will come to the store in six weeks.
So we cannot use that number, even though it is in the data. We only use
information that we already know today: the date, the store, and the promotions
and opening days that the company has planned.

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

This has two practical effects. We cannot divide by zero, so we cannot score
days with zero sales. Closed days are therefore removed from the data. For those
days we simply predict zero, and we do not need a model. And because the metric
squares the error, a few very bad days can carry a large part of the total.
That second point turned out to be the key to the whole project.

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

## How we test a change

The competition asks for six weeks that we cannot see, so our validation
windows are six weeks long too. We build four of them, one after the other,
going back in time from July 2015. Each window trains on every day before it
starts. A random split would let the model see days that come after the days
it predicts, and the score would be far too good.

The four windows are not equally hard. The two spring windows contain carnival
and Easter, and their error is larger. This is exactly why we use four windows
and not one. For the baseline, a single window would have reported 0.1449,
which is the friendliest of the four results.

XGBoost draws a random part of the rows and columns for each tree, so a single
run moves by a few thousandths, and one lucky seed can flatter a result. Every
accepted model is therefore the average of three seeds. **A change stays only
if it is better on every window by more than the seed noise.**

From notebook 07 on, a new idea is first screened with one seed on all four
windows. Only an idea that wins on every window is run again with three seeds,
and nothing is accepted before that. This cuts the training time by more than
half, and the bar for accepting a change stays where it was.

## Results

Every model is measured on the four validation windows described above.

| Window ends | Baseline | XGBoost | Per segment | + closed days and Easter |
|---|---|---|---|---|
| 2015-03-27 | 0.1877 | 0.1795 | 0.1759 | 0.1657 |
| 2015-05-08 | 0.1746 | 0.1523 | 0.1441 | 0.0998 |
| 2015-06-19 | 0.1584 | 0.1305 | 0.1260 | 0.0999 |
| 2015-07-31 | 0.1449 | 0.1364 | 0.1345 | 0.1102 |
| **Average** | **0.1664** | **0.1497** | **0.1451** | **0.1189** |

A lower number is better. The baseline is simple: for each store, weekday and
promotion state, it predicts the median sales of the past. The XGBoost columns
are the average of three random seeds, and every model wins on every window.

The last column is the final model: one XGBoost per store group, trained on
the logarithm of the sales, with the closed-day columns and the distance to
Easter. Three decisions did the work.

**Training on the logarithm of the sales** is worth about 0.008. RMSPE reads
every error as a percentage, and the logarithm makes the training goal read
them the same way. Without it the model even loses to the baseline on the
hardest window.

**One model per store group** is worth another 0.005. We describe each store by
ratios, such as how much a promotion helps it, and put the stores into four
behaviour groups. Four models, one for each group, beat a single model on every
window. Handing the group to a single model as a feature does almost nothing,
so the grouping has to change the shape of the model, not only the list of
features.

**Showing the model the closed days around each date** is worth 0.024, the
largest single gain of the project, and the error analysis found it. Six small
columns say whether the shop was open yesterday, whether it will be open
tomorrow, and how far the closest state holiday is. All of them are known six
weeks in advance. One more small feature, the distance to Easter, adds 0.002
on top - almost all of it on the carnival and Easter windows, which is exactly
what it was built for. Notebook `06_features.ipynb` shows the test, and also a
check that the gain lands on the closure days and not somewhere surprising.

## Where the model fails

The notebook `05_error_analysis.ipynb` runs the segmented model again, keeps
every scored row, and asks where the error sits. This notebook decided what
was built next.

**The error is in the days, not in the stores.** This is the main result, and it
was not what we expected. The worst 10 percent of the stores carry 33 percent of
the error, which is not much more than their share. But there are only 168
validation days, and the **worst ten of them carry 36.4 percent of the error**. A
store with a large error in one window is mostly fine in the next one: the
correlation between the windows is only +0.08 to +0.31.

**Two days explain the hardest window.** The 16th and 17th of February 2015 are
5.5 percent of the rows of that window and carry 47 percent of its error.
Without them the window scores 0.1312 instead of 0.1759, which would make it the
second easiest of the four. Those are the German carnival days.

**The model cannot see the closed days around it.** A day that follows a closed
day scores 0.187, and a day before one scores 0.171, against 0.123 for a normal
day. Together they are 38 percent of the rows. The reason is simple: closed days
are removed from our data, so the model knows that today is a holiday but not
that tomorrow is one. The days around Easter, Ascension Day and Whit Monday are
all predicted 17 to 28 percent too low, because people buy before the shop
closes. This finding became the closed-day columns of the final model.

**Easter moves, and the model cannot follow it.** Carnival is always 48 days
before Easter, so it fell on the 11th of February in 2013, the 3rd of March in
2014 and the 16th of February in 2015. The model reads a date as a month, a day
and a week number, so it cannot line those years up. The carnival week scores
0.335 and the week before Easter 0.213, against 0.09 to 0.15 for a normal week.
This finding became the distance to Easter.

The notebook also warns against a mistake that was easy to make. Monday, "the
day after a closed day" and "the first day of a promotion" all look like strong
findings on their own, but they are almost the same rows: 23,829 of the 24,543
Monday rows follow a closed Sunday. Counting them as three features would count
one effect three times.

## What did not work, and why

Half of the ideas in this project were tested and rejected. They stay in the
notebooks with their numbers, because a rejection with a reason is a result
too.

**A clear pattern is not a feature.** Notebook 03 found two clean patterns in
the raw data, built a feature for each, and measured no gain at all. The model
already knew them through the store averages. This became the rule for every
later idea: a pattern in the data is a reason to test, not a reason to keep.

**The days since a renovated store came back.** The error analysis found that
a store scores 0.265 in its first month back. The feature counts the days
since a long closure, and a second version also computes the store averages
from the new history only. Both versions lose. The column improves the first
month back, from 0.31 to 0.28, but that is 47 rows. Every later group of days
is worse, and so are the 134,834 rows of the stores that never closed for long.
The problem is real, but it is too narrow for a feature that every tree can
see. Notebook `07_reopening_and_month.ipynb` has the test.

**The start of the month.** Days 1 to 5 carry a larger error than days 6 to 10.
We built a flag for the first five days and a countdown to the end of the
month, and the score did not move at all: 0.1211 against 0.1210. We expected
this before running it. The model already has the day of the month as a
number, and a tree can split on it whenever that helps.

**Tuning the model.** Notebook `08_tuning_and_calibration.ipynb` ranks eight
settings on the inner six weeks of the newest window, in about seven minutes.
The settings we had used since notebook 03 won: 0.1021 against 0.1033 for the
closest challenger, which takes two and a half times as long. Every setting
with a faster learning rate was at least 0.003 behind. The confirmation run
also taught us that the patience of the early stopping is a setting in its own
right: with 30 rounds instead of 50 the score was 0.002 worse on two windows.
The model keeps the settings that were measured properly.

**The correction for the logarithm.** The model learns the logarithm of the
sales, and undoing it leaves the predictions a little low. One multiplier,
learned on the six weeks before each window, should fix that. It does not: the
score gets worse by 0.0007. The reason is that the bias moves. On the oldest
window the model over-predicts by three percent, on the later ones it
under-predicts by one, and the next six weeks want a different factor than the
six weeks before them. Even the best possible factor, learned on the scored
rows themselves, would gain only 0.0012.

## From the experiment to the forecast

Notebook `09_forecast.ipynb` closes the project. It starts with the road
that led here - the order of the steps and why each came where it did, what a
backtest is, and the four jobs a hidden window does - and then it does the
work that the other notebooks left out.

**The saved file gives the number of the notebooks.** The final model lives
in `src/forecaster.py`: one object that holds the fitted store statistics,
the store groups, the four models and the calendar of opening days, and that
can be saved and loaded. Trained on the days before the last validation
window, saved, loaded and asked to predict that window, it scores 0.1083
against 0.1102 in notebook 06. The production path and the experiment are
the same code.

**The model has learned, not memorised.** On the days it trained on the
model scores 0.0895, on the hidden window 0.1083, and the baseline on that
window is 0.1449. The gap between training and hidden days is small, and both
are far below the baseline. The number of trees is chosen inside every fit by
early stopping, which is the main guard against memorising.

**Leakage, shown and not only claimed.** Every one of the 31 features is
listed with the place it comes from: the date, the facts about the store, the
plans of the company and the state, the opening plan, and statistics learned
from the training past. Then the experiment notebook 01 refused: with the
`Customers` column added, the hidden window scores 0.0620 instead of 0.1083.
That is what a leak looks like - a score no real forecast could reach.

**The forecast.** The final model learns from every day up to 31 July 2015
and scores the 41,088 rows of `test.csv`. The competition never published the
answers, so the forecast is checked for shape instead: closed days are zero,
the mean per weekday matches the last six weeks of training, and the level of
each store sits at a median of 1.002 times its recent past. The saved model
also scores a single new row, with one lesson worth knowing: the row must
come with the opening plan of the days around it, because the closed-day
columns need to know whether tomorrow is open. Alone, store 1 on 20 August
is predicted at 1,485; with its plan, at 4,496.

The notebook ends with what a company adds after this point - monitoring,
scheduled retraining, a model registry, serving - and why none of it is
needed here.

## Project structure

```
data/                 the raw CSV files (you download them, they are not shared here)
notebooks/
  01_eda.ipynb        data checks and exploration
  02_baseline.ipynb   the metric, the validation windows and the baseline
  03_xgboost.ipynb    the gradient boosting model and what it is worth
  04_segmentation.ipynb  grouping the stores, and one model per group
  05_error_analysis.ipynb  where the model fails, and what to build next
  06_features.ipynb   the closed days and Easter enter the model
  07_reopening_and_month.ipynb  the last two ideas of the list, tested and rejected
  08_tuning_and_calibration.ipynb  the settings of the model, and the correction for the logarithm
  09_forecast.ipynb   the road from experiment to forecast, the final model, and the test predictions
src/
  data_loader.py      reads the files, joins them, cleans them
  features.py         builds the features for the model
  metrics.py          RMSPE, and a way to split the error by group
  validation.py       cuts the data by time, in the shape of the real task
  model.py            XGBoost on the logarithm of the sales
  experiment.py       runs one approach on every fold and collects the scores
  segments.py         groups the stores by the way they react
  forecaster.py       the final model: train once, save, score new rows
models/               the saved model and its card (written by the fit command, not shared)
requirements.txt      the Python packages you need, with their versions
LICENSE               MIT
```

The notebooks tell the story in order, and each one ends with its key
findings. The code they share lives in `src/`.

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

The builder can also take a calendar with the closed days kept. From it it
builds the closed-day columns, such as `open_tomorrow` and the distance to the
closest state holiday. This is reference data, not something we learn: the
company plans its opening days in advance, and the competition provides them
for the test period too. The rejected features are in the same file, behind
flags that are off by default, so every notebook keeps reproducing.

`src/validation.py` cuts the data by date into the four windows described
above.

`src/metrics.py` holds RMSPE and a function that splits the error by any
column, for example by store type or by month. That breakdown tells us where
the model is weak.

`src/model.py` trains XGBoost on the logarithm of the sales, and turns the
prediction back at the end. It also chooses the number of trees on a small
inner window taken from the end of the training data, so that the validation
window stays untouched. When only the random seed changes, the tree count
from the first seed can be passed in and the inner window is skipped.

`src/experiment.py` runs one approach on all four windows and collects the
scores. It asks for a function that trains on the rows of a single fold, which
keeps the fitting inside the fold where it belongs.

`src/segments.py` puts every store into one of four groups. It describes a
store by ratios, such as how much a promotion helps it, so that the groups
follow the behaviour of a store and not its size.

`src/forecaster.py` is the model that stays. It holds every fitted part - the
store statistics, the groups, the four models - and the calendar of opening
days, in one object that can be saved and loaded. It can be used from Python
or from the command line, and the file it writes comes with a small JSON card:
the training period, the tree counts, the code version and the package
versions.

## What is not in this project

Some things were left out on purpose.

- **One model per store.** With 1,115 stores that is 1,115 models. A model
  with store features is both better and far cheaper here.
- **Deep learning.** On 800,000 rows of tabular data, gradient boosting is
  hard to beat and much faster to work with.
- **Ensembles and stacking.** They only make sense after a single model has
  stopped improving, and they slow down every experiment that follows.
- **A Kaggle submission.** The competition is closed. The four validation
  windows are built in the shape of its test set, and that is what the
  project is judged on.

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
5. To train the final model and score a file without a notebook:
   ```
   python -m src.forecaster fit
   python -m src.forecaster predict data/test.csv --out data/predictions_test.csv
   ```
   The first command writes `models/forecaster.pkl` with its model card. The
   second reads any file in the shape of `test.csv` and writes it back with a
   `prediction` column.

The notebooks are saved with their outputs, so you can read them without
running anything. If you do run them, the ones that train models take between
a quarter of an hour and an hour each on a normal laptop.
