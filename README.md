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
  05_error_analysis.ipynb  where the model fails, and what to build next
  06_features.ipynb   the closed days and Easter enter the model
  07_reopening_and_month.ipynb  the last two ideas of the list, tested and rejected
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

The builder can also take a calendar with the closed days kept. From it it
builds the closed-day columns, such as `open_tomorrow` and the distance to the
closest state holiday. This is reference data, not something we learn: the
company plans its opening days in advance, and the competition provides them
for the test period too.

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
window stays untouched. When only the random seed changes, the tree count
from the first seed can be passed in and the inner window is skipped.

`src/experiment.py` runs one approach on all four windows and collects the
scores. It asks for a function that trains on the rows of a single fold, which
keeps the fitting inside the fold where it belongs.

`src/segments.py` puts every store into one of four groups. It describes a
store by ratios, such as how much a promotion helps it, so that the groups
follow the behaviour of a store and not its size.

## Results so far

Every model is measured on the four validation windows described above.

| Window ends | Baseline | XGBoost | Per segment | + closed days and Easter |
|---|---|---|---|---|
| 2015-03-27 | 0.1877 | 0.1795 | 0.1759 | 0.1657 |
| 2015-05-08 | 0.1746 | 0.1523 | 0.1441 | 0.0998 |
| 2015-06-19 | 0.1584 | 0.1305 | 0.1260 | 0.0999 |
| 2015-07-31 | 0.1449 | 0.1364 | 0.1345 | 0.1102 |
| **Average** | **0.1664** | **0.1497** | **0.1451** | **0.1189** |

A lower number is better. The baseline is simple: for each store, weekday and
promotion state, it predicts the median sales of the past.

The XGBoost columns are the average of three random seeds. The model draws
80 percent of the columns for each tree, so a single run moves by a few
thousandths and one seed alone can flatter a result. Every model wins on every
window, which is our rule for accepting a change.

Three decisions did the work.

**Training on the logarithm of the sales** is worth about 0.008. RMSPE reads
every error as a percentage, and the logarithm makes the training goal read
them the same way. Without it the model even loses to the baseline on the
hardest window.

**One model per store group** is worth another 0.005. Four models, one for each
behaviour group, beat a single model on every window. Handing the group to a
single model as a feature does almost nothing, so the grouping has to change
the shape of the model, not only the list of features.

**Showing the model the closed days around each date** is worth 0.024, the
largest single gain of the project, and the error analysis found it. Six small
columns say whether the shop was open yesterday, whether it will be open
tomorrow, and how far the closest state holiday is. All of them are known six
weeks in advance. One more small feature, the distance to Easter, adds 0.002
on top - almost all of it on the carnival and Easter windows, which is exactly
what it was built for. Notebook `06_features.ipynb` shows the test, and also a
check that the gain lands on the closure days and not somewhere surprising.

The four windows are not equally hard, and the older ones have a larger error.
This is exactly why we use four windows and not one. For the baseline, a single
window would have reported 0.1449, which is the friendliest of the four
results.

The notebook `02_baseline.ipynb` tests three reasons for that difference. It is
not the length of the training history: when every window gets the same
training data, the trend stays. It is not the renovated stores either, because
they change the score by 0.0024 at most. What is left is the season. The spring
windows are simply harder to predict than the summer windows.

## Where the model fails

The notebook `05_error_analysis.ipynb` runs the segmented model again, keeps
every scored row, and asks where the error sits. It also trains the global model
on the same rows, so every table can show whether the split helped that group.

**The error is in the days, not in the stores.** This is the main result, and it
was not what we expected. The worst 10 percent of the stores carry 33 percent of
the error, which is not much more than their share. But there are only 168
validation days, and the **worst ten of them carry 36.4 percent of the error**. A
store with a large error in one window is mostly fine in the next one: the
correlation between the windows is only +0.08 to +0.31.

**Two days explain the hardest window.** The 16th and 17th of February 2015 are
5.5 percent of the rows of that window and carry 47 percent of its error.
Without them the window scores 0.1312 instead of 0.1759, which would make it the
second easiest of the four. Those are the German carnival days. Notebook 02 had
called the difference between the windows "the season", and that answer is now
replaced by a better one.

**The model cannot see the closed days around it.** A day that follows a closed
day scores 0.187, and a day before one scores 0.171, against 0.123 for a normal
day. Together they are 38 percent of the rows. The reason is simple: closed days
are removed from our data, so the model knows that today is a holiday but not
that tomorrow is one. The days around Easter, Ascension Day and Whit Monday are
all predicted 17 to 28 percent too low, because people buy before the shop
closes.

**Easter moves, and the model cannot follow it.** Carnival is always 48 days
before Easter, so it fell on the 11th of February in 2013, the 3rd of March in
2014 and the 16th of February in 2015. The model reads a date as a month, a day
and a week number, so it cannot line those years up. The carnival week scores
0.335 and the week before Easter 0.213, against 0.09 to 0.15 for a normal week.

**One more small but clear group.** A month after a renovated store comes back,
the error is still 0.265. Thirty days later it is 0.127. The problem is narrow
and it ends by itself.

The notebook also warns against a mistake that was easy to make. Monday, "the
day after a closed day" and "the first day of a promotion" all look like strong
findings on their own, but they are almost the same rows: 23,829 of the 24,543
Monday rows follow a closed Sunday. Counting them as three features would count
one effect three times.

## Two ideas that did not work

The error analysis list had two items left, and notebook
`07_reopening_and_month.ipynb` tests them. Both are rejected.

**The days since a renovated store came back.** The error analysis had
found that a store scores 0.265 in its first month back. The feature counts
the days since a long closure, and a second version also computes the store
averages from the new history only. Both versions lose. The column improves
the first month back, from 0.31 to 0.28, but that is 47 rows. Every later
group of days is worse, and so are the 134,834 rows of the stores that never
closed for long. On the oldest window, where the reopened stores have only a
few weeks of new history, the loss is 0.005 with the column and 0.010 with the
rewritten averages. The problem is real, but it is too narrow for a feature
that every tree can see.

**The start of the month.** Days 1 to 5 carry a larger error than days 6 to
10. We built a flag for the first five days and a countdown to the end of the
month, and the score did not move at all: 0.1211 against 0.1210. We expected
this before running it. The model already has the day of the month as a
number, and a tree can split on it whenever that helps. Notebook 03 had the
same lesson.

The base in that notebook is trained with a higher learning rate, so that a
full experiment takes a quarter of an hour instead of an hour. This costs
0.002, which is why the base reads 0.1210 there and 0.1189 in the table
above. Every arm of the experiment pays the same price, so the comparison is
fair, and the tuning step will set the learning rate properly.

The notebook also changes how we test from now on. Three seeds are the right
price to accept a feature, but a high price to reject one: if an idea is not
better on the four windows with one seed, two more seeds will not save it. So
a new idea is first **screened** with one seed on all four windows. Only an
idea that wins on every window by more than the seed noise is run again with
three seeds, and nothing is accepted before that. The base model always runs
with three seeds, which is how we know the seed noise. This cuts the training
time of a notebook by more than half, and the bar for accepting a change stays
where it was.

## Next steps

The metric, the validation setup, the store groups, the error analysis and
every feature family it asked for are ready. Two steps remain:

1. Tune the model parameters, now against a much stronger model.
2. Correct the small bias that the logarithm leaves behind. It is about 1.3
   percent and it is the same in every group, so one correction for all stores
   should be enough.

Both have to prove themselves the same way as everything before them: four
windows, a win on every window, and three seeds before anything is accepted.
Notebook 03 is the reminder. A pattern can be very clear in the data and still
give the model nothing.

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
