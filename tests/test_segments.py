import polars as pl

from src.segments import StoreSegmenter


def test_every_store_gets_exactly_one_segment(train):
    segmenter = StoreSegmenter(n_segments=4).fit(train)
    assignment = segmenter.transform(train).group_by("Store").agg(pl.col("segment").n_unique())
    assert (assignment["segment"] == 1).all()
    segments = segmenter.transform(train)["segment"]
    assert segments.null_count() == 0
    assert set(segments.unique().to_list()) <= {1, 2, 3, 4}


def test_unknown_store_gets_the_default_segment(train):
    segmenter = StoreSegmenter(n_segments=4).fit(train)
    stranger = train.head(3).with_columns(pl.lit(9999).alias("Store"))
    assert (segmenter.transform(stranger)["segment"] == segmenter.default_segment_).all()
