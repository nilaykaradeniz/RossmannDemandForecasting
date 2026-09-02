# Rossmann Demand Forecasting

![tests](https://github.com/nilaykaradeniz/RossmannDemandForecasting/actions/workflows/tests.yml/badge.svg)

Rossmann is a large drugstore chain in Germany. This project predicts how much
each of its 1,115 stores will sell on each day, six weeks into the future, so
that a store can order the right stock and plan its staff.

## In short

- **The final model scores 0.1189 RMSPE** on four time-ordered validation
  windows. A strong baseline scores 0.1664, so the model is 29 percent better.
  For a typical store that is a median miss of about 400 units on a day of
  6,400, with no lean to either side.
- **The error analysis did the work.** The largest single gain, 0.024, came
  from six small columns about closed days. We built them because a notebook
  showed where the model was failing, not because we guessed.
- **Every change had to prove itself** on four windows, with three random
  seeds, and win on every window. Ideas that did not pass stay in the
  notebooks with their numbers.
- **Tuning and calibration came last, and added nothing.** Every gain came
  from the data and the features.
- **No leakage, shown and not only claimed.** Every feature is listed with
  the place it comes from, and the one column that would leak is added once
  to show what a leak looks like.
- **The model ships.** It is trained once, saved with a model card, and used
  from a file to forecast the test period, with an interval, and to score
  single new rows. It refuses a row that comes without its opening plan.
- **The open questions were asked and answered.** A senior review of this
  repository produced a list of questions; notebook 10 measures each one.

## The problem

We predict daily sales for each store, six weeks into the future.

Six weeks is a long time, and this changes what we are allowed to use. We do
not know how many customers will come to the store in six weeks, so we cannot
use that number, even though it is in the data. We only use what the company
already knows today: the date, the store, and the promotions and opening days
it has planned.

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

We use RMSPE (Root Mean Square Percentage Error). It works with percentages,
so an error of 500 units is large for a small store and small for a big one,
and the metric sees both in a fair way.

Two consequences shaped the project. We cannot divide by zero, so closed days
are removed from the data and predicted as zero without a model. And because
the metric squares the error, a few very bad days can carry a large part of
the total. That second point turned out to be the key to the whole project.

## What we found in the data

The notebook `01_eda.ipynb` explores the data. The short version:

- **The data is clean**, with one placeholder value (a competitor that opened
  in 1900).
- **Sales follow a weekly pattern**, Monday strongest, and almost every store
  closes on Sunday. December is the best month; there is no growth trend over
  the two and a half years.
- **Promotions work well**, and the company plans them in advance, so they are
  a feature we may use.
- **About 180 stores disappear for six months in 2014**, closed for
  renovation. Their history is shorter than the others'.
- **The `Customers` column must not be used.** It is almost the same fact as
  the sales, and nobody knows it six weeks ahead.

## How we test a change

The competition asks for six weeks that we cannot see, so our validation
windows are six weeks long too. We build four of them, one after the other,
going back in time from July 2015. Each window trains on every day before it
starts. A random split would let the model see days that come after the days
it predicts, and the score would be far too good.

The four windows are not equally hard. The two spring windows contain carnival
and Easter, and their error is larger. This is exactly why we use four windows
and not one: for the baseline, a single window would have reported 0.1449,
the friendliest of the four results.

XGBoost draws a random part of the rows and columns for each tree, so a single
run moves by a few thousandths, and one lucky seed can flatter a result. Every
accepted model is therefore the average of three seeds. **A change stays only
if it is better on every window by more than the seed noise.** From notebook
07 on, a new idea is first screened with one seed on all four windows, and
only a winner is run again with three.

## Results

| Window ends | Baseline | XGBoost | Per segment | + closed days and Easter |
|---|---|---|---|---|
| 2015-03-27 | 0.1877 | 0.1795 | 0.1759 | 0.1657 |
| 2015-05-08 | 0.1746 | 0.1523 | 0.1441 | 0.0998 |
| 2015-06-19 | 0.1584 | 0.1305 | 0.1260 | 0.0999 |
| 2015-07-31 | 0.1449 | 0.1364 | 0.1345 | 0.1102 |
| **Average** | **0.1664** | **0.1497** | **0.1451** | **0.1189** |

A lower number is better. The baseline predicts, for each store, weekday and
promotion state, the median sales of the past. The XGBoost columns are the
average of three random seeds, and every model wins on every window.

The last column is the final model: one XGBoost per store group, trained on
the logarithm of the sales, with the closed-day columns and the distance to
Easter. Three decisions did the work.

**Training on the logarithm of the sales** is worth about 0.008. RMSPE reads
every error as a percentage, and the logarithm makes the training goal read
them the same way. Without it the model even loses to the baseline on the
hardest window.

**One model per store group** is worth another 0.005. We describe each store by
ratios, such as how much a promotion helps it, and put the stores into four
behaviour groups. Four models beat a single model on every window; handing
the group to a single model as a feature does almost nothing.

**Showing the model the closed days around each date** is worth 0.024, the
largest single gain of the project, and the error analysis found it. Six small
columns say whether the shop was open yesterday, whether it will be open
tomorrow, and how far the closest state holiday is. All of them are known six
weeks in advance. The distance to Easter adds 0.002 on top, almost all of it
on the carnival and Easter windows. Notebook `06_features.ipynb` has the
test.

## What the error means for the business

On the last validation window, a typical open day sells 6,400 units and the
median miss is 400 units, or 6 percent. Half of the days are within 400 units,
nine in ten within 1,200. The forecast is too high on 48 percent of the days
and too low on 52 percent, and over the six weeks it sums to 0.7 percent below
the real total, so it does not lean to one side.

The error does not grow with the distance. The first week ahead scores 0.094,
the sixth 0.125, and the weeks between move up and down with the calendar,
not with the horizon. The reason is that the model has no memory of the last
few days: every feature is known in advance, so week six is as easy as week
one. The price of that design is that the model cannot react to a shock that
started yesterday.

Each forecast can come with an 80 percent interval, built from the spread of
the model's own recent misses. On the last window it held 75 percent of the
real sales, a little less than promised; notebook 10 says why.

## Where the model fails, and what that decided

The notebook `05_error_analysis.ipynb` keeps every scored row of the segmented
model and asks where the error sits. Three findings decided the rest of the
project.

**The error is in the days, not in the stores.** The worst 10 percent of the
stores carry 33 percent of the error, barely more than their share. But the
worst ten of the 168 validation days carry 36 percent, and two carnival days
alone carry 47 percent of the hardest window's error.

**The model cannot see the closed days around it.** A day after a closed day
scores 0.187 and a day before one 0.171, against 0.123 for a normal day. The
closed days are removed from the data, so the model knows that today is a
holiday but not that tomorrow is one. This became the closed-day columns.

**Easter moves, and the model cannot follow it.** Carnival is always 48 days
before Easter, which fell on three different dates in the three years. The
model reads a date as a month and a day, so it cannot line the years up. This
became the distance to Easter.

## What did not work, and why

Half of the ideas in this project were tested and rejected. They stay in the
notebooks with their numbers, because a rejection with a reason is a result
too.

**A clear pattern is not a feature.** Notebook 03 found two clean patterns in
the raw data, built a feature for each, and measured no gain at all. The model
already knew them through the store averages.

**The days since a renovated store came back** (notebook 07). The column
improves the first month back, from 0.31 to 0.28, but that is 47 rows. Every
later group of days is worse, and so are the 134,834 rows of stores that
never closed for long. A real problem, too narrow for a feature every tree
can see.

**The start of the month** (notebook 07). A flag for days 1 to 5 and a
countdown to the end of the month: 0.1211 against 0.1210. The model already
has the day of the month as a number.

**Tuning** (notebook 08). Eight settings ranked on the inner six weeks of the
newest window; the settings used since notebook 03 won, and every faster
learning rate was at least 0.003 behind.

**The correction for the logarithm** (notebook 08). One multiplier, learned on
the six weeks before each window, makes the score worse by 0.0007, because
the bias moves from one six-week block to the next. Even the best possible
factor would gain only 0.0012.

## From the experiment to the forecast

Notebook `09_forecast.ipynb` starts with the road that led here - the order
of the steps and why each came where it did, what a backtest is, and the four
jobs a hidden window does - and then does the work the other notebooks left
out.

**The saved file gives the number of the notebooks.** The final model lives
in `src/forecaster.py`: one object that holds the fitted store statistics,
the store groups, the four models and the calendar of opening days. Trained
on the days before the last validation window, saved, loaded and asked to
predict that window, it scores 0.1083 against 0.1102 in notebook 06.

**The model has learned, not memorised.** On the days it trained on the
model scores 0.0895, on the hidden window 0.1083, and the baseline on that
window is 0.1449.

**Leakage, shown and not only claimed.** Every one of the 31 features is
listed with the place it comes from. Then the experiment notebook 01 refused:
with the `Customers` column added, the hidden window scores 0.0620. That is
what a leak looks like.

**The forecast.** The final model learns from every day up to 31 July 2015
and scores the 41,088 rows of `test.csv`. The competition never published the
answers, so the forecast is checked for shape: closed days at zero, weekday
means matching the last six weeks, store levels at a median of 1.002 times
their recent past. A single new row is scored too, with one lesson: it must
come with the opening plan of the days around it, and `predict` refuses it
otherwise. The notebook ends with what a company adds after this point -
monitoring with thresholds, retraining with a challenger, a model registry -
and why the loop itself is not here.

## The questions a reviewer would ask

A senior review of this repository produced a list of questions. Notebook
`10_review_questions.ipynb` answers each one on the last validation window,
with one seed. The short answers:

- **Why gradient boosting and not a time-series model?** A second baseline of
  the time-series kind, the same weekday one year ago, scores 0.161, worse
  than the median baseline at 0.145; the model scores 0.108. With 1,115 short
  series that share one calendar and one set of promotions, a model that
  learns across the stores beats a model per store, and it takes the
  promotion plan and the opening plan as inputs, which a classical series
  model cannot.
- **Why the logarithm, and not an objective closer to RMSPE?** The exact objective - raw sales with the weight `1 / sales squared` - scores 0.1145 against 0.1083 for the logarithm. The weights hand the trees to the small stores; the logarithm gives every store the same voice.
- **Why four segments?** Three score 0.1089, four 0.1083, six 0.1112. Three
  and four are within the seed noise; six splits the stores too thin.
- **Notebook 08 left 0.002 unexplained. Patience or tree reuse?** Patience.
  Waiting 30 rounds instead of 50 stops two of the four groups far too early
  and costs 0.006 on this window; reusing the tree count across seeds costs
  0.0003, nothing.
- **One seed in production, three in the backtest?** Averaging the
  predictions of three seeds scores 0.1084 against 0.1096 for the mean of
  their single scores - one thousandth, at three times the file. The option
  exists (`seeds=`), the default stays one.
- **How sure is the forecast?** The 80 percent interval, built from each
  group's misses on its inner window, is 23 percent of the prediction wide
  and held 75 percent of the real sales six weeks later. The spread of six
  weeks ago is a little narrower than the spread of today.
- **Why is the model file 75 MB?** 49 MB are the four boosters, about a
  thousand trees each; 25 MB are the calendar of opening days, kept so that
  any date can be scored, not only the future.
- **What is distinctive about the closed-day family?** Not the feature; the
  road to it, which predicted the gain before the columns were built and then
  said no to two other features and to the tuning.
- **The weekly retraining gives a worse model - then what?** It is not
  promoted. It is scored on the last window next to the model in production
  and replaces it only if it is not worse by more than the seed noise.
- **How is drift noticed?** `evaluate` scores every week against the real
  sales; the backtest's windows scored between 0.10 and 0.17, so one week
  above 0.17 is a warning and two in a row are an alarm. The shape checks of
  notebook 09 run on the new rows before they are scored.

## What I would do with more time

- **Lag features, with a short horizon.** The model has no memory of the last
  days on purpose. A second model for the first week, with lags, would react
  to shocks; the six-week model would stay for the plan.
- **An asymmetric loss.** Running out of stock and holding too much stock do
  not cost the same. With the two costs from the business, a quantile model
  would forecast the level that minimises the expected cost, not the median.
- **Calibrated intervals.** The 80 percent interval holds 75 percent. A
  wider quantile, or ratios updated every week, would close the gap.
- **The two carnival days.** They carry 47 percent of the hardest window's
  error. A feature for the carnival Monday and Tuesday specifically was not
  tried.

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
  10_review_questions.ipynb  the questions a reviewer would ask, each one measured
src/
  data_loader.py      reads the files, joins them, cleans them
  features.py         builds the features for the model
  metrics.py          RMSPE, and a way to split the error by group
  validation.py       cuts the data by time, in the shape of the real task
  model.py            XGBoost on the logarithm of the sales
  experiment.py       runs one approach on every fold and collects the scores
  segments.py         groups the stores by the way they react
  forecaster.py       the final model: train once, save, score new rows, evaluate later
tests/                unit tests on made-up data, run by pytest and by GitHub Actions
models/               the saved model and its card (written by the fit command, not shared)
requirements.txt      the Python packages you need, with their versions
LICENSE               MIT
```

The notebooks tell the story in order, and each one ends with its key
findings. The code they share lives in `src/`.

`src/features.py` builds the features with the `fit` and `transform` pattern:
`fit` learns the store statistics from the training data only, `transform`
builds the columns known in advance and joins the learned values. This is
what protects the model from leakage. The closed-day columns come from a
calendar of opening days, which is reference data, not something learned.

`src/validation.py` cuts the data by date into the four windows. `src/model.py`
trains XGBoost on the logarithm of the sales and chooses the number of trees
on an inner window taken from the end of the training data, so that the
validation window stays untouched. `src/segments.py` puts every store into one
of four behaviour groups. `src/forecaster.py` holds all of it in one object
that can be saved, loaded and used from the command line, and writes a JSON
card next to the model file.

`tests/` holds 37 unit tests. The real data cannot be shared, so they build a
small made-up table with the same shape and run every part of `src` on it in
seconds. GitHub Actions runs them on every push; notebook 09 tells what the
first run taught.

## What is not in this project

- **One model per store.** With 1,115 stores that is 1,115 models. A model
  with store features is both better and far cheaper here.
- **Deep learning.** On 800,000 rows of tabular data, gradient boosting is
  hard to beat and much faster to work with.
- **Ensembles and stacking.** They only make sense after a single model has
  stopped improving, and they slow down every experiment that follows.
- **A Kaggle submission.** The competition is closed. The four validation
  windows are built in the shape of its test set, and that is what the
  project is judged on.
- **MLOps tooling.** The loop a company would build around this model is
  described in notebook 09; the tools for it are not needed on one machine.

## How to run

The project runs on Python 3.10, with the package versions in
`requirements.txt`.

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
4. Open the notebooks in `notebooks/` and run them from top to bottom. They
   are saved with their outputs, so you can also read them without running
   anything. The ones that train models take between five minutes and an hour
   each on a normal laptop.
5. To train the final model and score a file without a notebook:
   ```
   python -m src.forecaster fit
   python -m src.forecaster predict data/test.csv --out data/predictions_test.csv --interval 0.8
   ```
6. When the real sales of a forecast period are known, score the forecast:
   ```
   python -m src.forecaster evaluate actual_sales.csv data/predictions_test.csv --threshold 0.17
   ```
7. To run the tests, which need no data:
   ```
   pytest
   ```
