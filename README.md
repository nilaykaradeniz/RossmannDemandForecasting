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

## The problem and the data

We predict daily sales for each store, six weeks into the future. Six weeks
is a long time, and this changes what we are allowed to use. We do not know
how many customers will come to the store in six weeks, so we cannot use that
number, even though it is in the data. We only use what the company already
knows today: the date, the store, and the promotions and opening days it has
planned.

The data comes from a Kaggle competition. It has about one million rows. Each
row is one store on one day.

| File | What it contains |
|---|---|
| `train.csv` | Daily sales of 1,115 stores, from January 2013 to July 2015 |
| `store.csv` | Facts about each store: its type, its product range, its competitors |
| `test.csv` | The days we must predict: August and September 2015 |

The CSV files are **not** in this repository. The Kaggle rules do not allow us
to share them. Please download them yourself. The steps are in the last
section. Notebook `01_eda.ipynb` explores the data.

## How we measure the error

We use RMSPE (Root Mean Square Percentage Error). It works with percentages,
so an error of 500 units is large for a small store and small for a big one,
and the metric sees both in a fair way.

Two consequences shaped the project. We cannot divide by zero, so closed days
are removed from the data and predicted as zero without a model. And because
the metric squares the error, a few very bad days can carry a large part of
the total. That second point turned out to be the key to the whole project.

## How we test a change

The competition asks for six weeks that we cannot see, so our validation
windows are six weeks long too. We build four of them, one after the other,
going back in time from July 2015. Each window trains on every day before it
starts. A random split would let the model see days that come after the days
it predicts, and the score would be far too good. The test file itself runs
48 days, not 42; the model has no memory of the last days, so the extra six
days are scored like any other day.

The four windows are not equally hard. The two spring windows contain carnival
and Easter, and their error is larger. This is why we use four windows and
not one: for the baseline, a single window would have reported 0.1449, the
friendliest of the four results.

XGBoost draws a random part of the rows and columns for each tree, so a single
run moves by a few thousandths. Every accepted model is therefore the average
of three seeds. **A change stays only if it is better on every window by more
than the seed noise.**

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

The last column is the final model. Three decisions did the work:

- **Training on the logarithm of the sales** (notebook 03) is worth about
  0.008. The logarithm makes the training goal read every error as a
  percentage, the way RMSPE does.
- **One model per store group** (notebook 04) is worth another 0.005. Stores
  are described by ratios, such as how much a promotion helps them, and put
  into four behaviour groups.
- **Showing the model the closed days around each date** (notebooks 05 and
  06) is worth 0.024, the largest gain of the project. The error analysis
  found that a day after a closed day scored 0.187 against 0.123 for a normal
  day: the closed days are removed from the data, so the model knew that
  today is a holiday but not that tomorrow is one. Six columns about the
  neighbouring days fixed that, and the distance to Easter added 0.002 on
  the carnival and Easter windows.

Four ideas were tested and rejected, and stay in the notebooks with their
numbers: two clean patterns from the raw data that the store averages already
covered (notebook 03), the days since a renovated store came back and the
start of the month (notebook 07), eight tuning settings that all lost to the
settings in use (notebook 08), and a correction for the logarithm that made
the score worse because the bias moves from one six-week block to the next
(notebook 08).

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

## From the experiment to the forecast

Notebook `09_forecast.ipynb` turns the experiment into a model that can be
used. The final model lives in `src/forecaster.py`: one object that holds the
fitted store statistics, the store groups, the four models and the calendar
of opening days. Three checks, on the last validation window:

- **The saved file gives the number of the notebooks.** Trained, saved,
  loaded and asked to predict the window, it scores 0.1083 against 0.1102 in
  notebook 06.
- **The model has learned, not memorised.** On the days it trained on it
  scores 0.0895, on the hidden window 0.1083, and the baseline is 0.1449.
- **Leakage, shown and not only claimed.** Every one of the 31 features is
  listed with the place it comes from. With the `Customers` column added, the
  window scores 0.0620. That is what a leak looks like.

The final model then learns from every day up to 31 July 2015 and scores the
41,088 rows of `test.csv`. The competition never published the answers, so the
forecast is checked for shape: closed days at zero, weekday means matching
the last six weeks, store levels at a median of 1.002 times their recent
past. A single new row must come with the opening plan of the days around it,
and `predict` refuses it otherwise. The notebook ends with what a company
adds after this point: monitoring with thresholds, retraining with a
challenger, a model registry.

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
- **Why six weeks?** Because the competition asked for six weeks. A company
  would start from the decision the forecast serves, and each decision has
  its own horizon: the lead time of an order is one or two weeks, a staff
  roster is published two to four weeks ahead, a budget looks at months.
  The backtest would follow the rhythm of that decision, with a new cut
  every week and many windows instead of four, and the error would be
  reported by week ahead. The likely end is two models: a short one with
  lags for the orders, and this one for the plan.
- **Why the logarithm, and not an objective closer to RMSPE?** The exact
  objective, raw sales with the weight `1 / sales squared`, scores 0.1145
  against 0.1083 for the logarithm. The weights hand the trees to the small
  stores; the logarithm gives every store the same voice.
- **Why four segments?** Three score 0.1089, four 0.1083, six 0.1112. Three
  and four are within the seed noise; six splits the stores too thin.
- **Notebook 08 left 0.002 unexplained. Patience or tree reuse?** Patience.
  Waiting 30 rounds instead of 50 stops two of the four groups far too early
  and costs 0.006 on this window; reusing the tree count across seeds costs
  0.0003, nothing.
- **One seed in production, three in the backtest?** Averaging the
  predictions of three seeds scores 0.1084 against 0.1096 for the mean of
  their single scores: one thousandth, at three times the file. The option
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
- **The weekly retraining gives a worse model, then what?** It is not
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
  features.py         builds the features: fit learns from the training data only, transform applies
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
