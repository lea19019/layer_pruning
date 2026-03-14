"""Tests for src/data_prep/filter.py and src/data_prep/split.py.

Only tests functions that do not require external models (fastText,
sentence-transformers). Those are: dedup, filter_length, load_raw_pairs,
save_pairs, split_data.
"""

from pathlib import Path

from src.data_prep.filter import dedup, filter_length, load_raw_pairs, save_pairs
from src.data_prep.split import load_parallel, split_data


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------

class TestDedup:
    def test_removes_exact_duplicates(self):
        pairs = [("a", "b"), ("c", "d"), ("a", "b"), ("e", "f"), ("c", "d")]
        result = dedup(pairs)
        assert len(result) == 3
        assert result == [("a", "b"), ("c", "d"), ("e", "f")]

    def test_no_duplicates(self):
        pairs = [("a", "b"), ("c", "d")]
        result = dedup(pairs)
        assert result == pairs

    def test_empty_input(self):
        assert dedup([]) == []

    def test_all_duplicates(self):
        pairs = [("x", "y")] * 10
        assert len(dedup(pairs)) == 1

    def test_preserves_order(self):
        pairs = [("c", "d"), ("a", "b"), ("c", "d")]
        result = dedup(pairs)
        assert result == [("c", "d"), ("a", "b")]


# ---------------------------------------------------------------------------
# filter_length
# ---------------------------------------------------------------------------

class TestFilterLength:
    def test_passes_short_balanced(self):
        pairs = [("hello world", "hallo welt")]
        result = filter_length(pairs)
        assert len(result) == 1

    def test_rejects_too_long_source(self):
        long_src = " ".join(["word"] * 201)
        pairs = [(long_src, "short")]
        result = filter_length(pairs)
        assert len(result) == 0

    def test_rejects_too_long_target(self):
        long_tgt = " ".join(["wort"] * 201)
        pairs = [("short", long_tgt)]
        result = filter_length(pairs)
        assert len(result) == 0

    def test_rejects_extreme_ratio(self):
        # ratio = 10/1 = 10.0, exceeds 1.5
        pairs = [("word", " ".join(["w"] * 10))]
        result = filter_length(pairs)
        assert len(result) == 0

    def test_passes_acceptable_ratio(self):
        # ratio = 3/2 = 1.5, exactly at threshold -- should pass
        pairs = [("a b c", "x y")]
        result = filter_length(pairs)
        assert len(result) == 1

    def test_rejects_just_over_ratio(self):
        # ratio = 4/2 = 2.0, exceeds 1.5
        pairs = [("a b c d", "x y")]
        result = filter_length(pairs)
        assert len(result) == 0

    def test_rejects_empty_segments(self):
        pairs = [("", "something"), ("something", "")]
        result = filter_length(pairs)
        assert len(result) == 0

    def test_boundary_200_words(self):
        exactly_200 = " ".join(["w"] * 200)
        pairs = [(exactly_200, exactly_200)]
        result = filter_length(pairs)
        assert len(result) == 1

    def test_empty_input(self):
        assert filter_length([]) == []


# ---------------------------------------------------------------------------
# load_raw_pairs
# ---------------------------------------------------------------------------

class TestLoadRawPairs:
    def test_loads_tsv(self, sample_tsv):
        pairs = load_raw_pairs(sample_tsv)
        assert len(pairs) == 5
        assert pairs[0] == ("Ahoj svete", "Hallo Welt")

    def test_skips_short_rows(self, tmp_path):
        tsv = tmp_path / "bad.tsv"
        with open(tsv, "w") as f:
            f.write("only_one_column\n")
            f.write("good\tpair\n")
        pairs = load_raw_pairs(tsv)
        assert len(pairs) == 1

    def test_skips_empty_fields(self, tmp_path):
        tsv = tmp_path / "empty_fields.tsv"
        with open(tsv, "w") as f:
            f.write("\t\n")
            f.write("hello\tworld\n")
            f.write("  \tworld\n")
        pairs = load_raw_pairs(tsv)
        assert len(pairs) == 1

    def test_extra_columns_ignored(self, tmp_path):
        tsv = tmp_path / "extra.tsv"
        with open(tsv, "w") as f:
            f.write("a\tb\tc\td\n")
        pairs = load_raw_pairs(tsv)
        assert len(pairs) == 1
        assert pairs[0] == ("a", "b")


# ---------------------------------------------------------------------------
# save_pairs
# ---------------------------------------------------------------------------

class TestSavePairs:
    def test_saves_parallel_files(self, tmp_path, sample_pairs):
        output_dir = tmp_path / "output"
        src_path, tgt_path = save_pairs(sample_pairs, output_dir, prefix="test")

        assert src_path.exists()
        assert tgt_path.exists()
        assert src_path.name == "test.cs"
        assert tgt_path.name == "test.de"

        with open(src_path) as f:
            src_lines = f.read().splitlines()
        with open(tgt_path) as f:
            tgt_lines = f.read().splitlines()

        assert len(src_lines) == len(sample_pairs)
        assert len(tgt_lines) == len(sample_pairs)
        assert src_lines[0] == "Ahoj svete"
        assert tgt_lines[0] == "Hallo Welt"

    def test_creates_output_dir(self, tmp_path, sample_pairs):
        output_dir = tmp_path / "new" / "nested"
        save_pairs(sample_pairs, output_dir)
        assert output_dir.exists()

    def test_empty_pairs(self, tmp_path):
        output_dir = tmp_path / "empty"
        src_path, tgt_path = save_pairs([], output_dir, prefix="empty")
        with open(src_path) as f:
            assert f.read() == ""


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_basic_split(self, sample_pairs):
        train, test = split_data(sample_pairs, train_size=3, test_size=2, seed=42)
        assert len(test) == 2
        assert len(train) == 3
        # No overlap
        test_set = set(test)
        train_set = set(train)
        assert len(test_set & train_set) == 0

    def test_deterministic(self, sample_pairs):
        train1, test1 = split_data(sample_pairs, train_size=3, test_size=2, seed=42)
        train2, test2 = split_data(sample_pairs, train_size=3, test_size=2, seed=42)
        assert train1 == train2
        assert test1 == test2

    def test_different_seed_different_split(self, sample_pairs):
        train1, test1 = split_data(sample_pairs, train_size=2, test_size=2, seed=1)
        train2, test2 = split_data(sample_pairs, train_size=2, test_size=2, seed=2)
        # Very likely different (could theoretically be equal but extremely unlikely)
        assert train1 != train2 or test1 != test2

    def test_insufficient_data(self):
        pairs = [("a", "b"), ("c", "d"), ("e", "f")]
        train, test = split_data(pairs, train_size=100, test_size=1, seed=42)
        assert len(test) == 1
        assert len(train) == 2  # remaining after test

    def test_all_data_used_when_insufficient(self):
        pairs = [("a", "b"), ("c", "d")]
        train, test = split_data(pairs, train_size=100, test_size=1, seed=42)
        assert len(test) + len(train) == len(pairs)


# ---------------------------------------------------------------------------
# load_parallel
# ---------------------------------------------------------------------------

class TestLoadParallel:
    def test_loads_files(self, sample_parallel_files):
        src_path, tgt_path = sample_parallel_files
        pairs = load_parallel(src_path, tgt_path)
        assert len(pairs) == 5
        assert pairs[0] == ("Ahoj svete", "Hallo Welt")

    def test_pairs_aligned(self, sample_parallel_files):
        src_path, tgt_path = sample_parallel_files
        pairs = load_parallel(src_path, tgt_path)
        for src, tgt in pairs:
            assert isinstance(src, str)
            assert isinstance(tgt, str)
            assert len(src) > 0
            assert len(tgt) > 0
