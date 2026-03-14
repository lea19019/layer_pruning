"""Tests for src/distillation/train_kd.py -- specifically merge_datasets."""

from pathlib import Path

from src.distillation.train_kd import merge_datasets


class TestMergeDatasets:
    def _write_lines(self, path: Path, lines: list[str]):
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def test_merge_correct_count(self, tmp_path):
        a_src = tmp_path / "a.cs"
        a_tgt = tmp_path / "a.de"
        k_src = tmp_path / "k.cs"
        k_tgt = tmp_path / "k.de"

        self._write_lines(a_src, ["a1", "a2", "a3"])
        self._write_lines(a_tgt, ["A1", "A2", "A3"])
        self._write_lines(k_src, ["k1", "k2"])
        self._write_lines(k_tgt, ["K1", "K2"])

        out_dir = tmp_path / "merged"
        merged_src, merged_tgt = merge_datasets(a_src, a_tgt, k_src, k_tgt, out_dir)

        with open(merged_src) as f:
            src_lines = f.read().strip().splitlines()
        with open(merged_tgt) as f:
            tgt_lines = f.read().strip().splitlines()

        assert len(src_lines) == 5
        assert len(tgt_lines) == 5

    def test_merge_preserves_order(self, tmp_path):
        a_src = tmp_path / "a.cs"
        a_tgt = tmp_path / "a.de"
        k_src = tmp_path / "k.cs"
        k_tgt = tmp_path / "k.de"

        self._write_lines(a_src, ["a1", "a2"])
        self._write_lines(a_tgt, ["A1", "A2"])
        self._write_lines(k_src, ["k1"])
        self._write_lines(k_tgt, ["K1"])

        out_dir = tmp_path / "merged"
        merged_src, merged_tgt = merge_datasets(a_src, a_tgt, k_src, k_tgt, out_dir)

        with open(merged_src) as f:
            src_lines = f.read().strip().splitlines()
        # Authentic first, then KD
        assert src_lines == ["a1", "a2", "k1"]

    def test_creates_output_dir(self, tmp_path):
        a_src = tmp_path / "a.cs"
        a_tgt = tmp_path / "a.de"
        k_src = tmp_path / "k.cs"
        k_tgt = tmp_path / "k.de"

        self._write_lines(a_src, ["x"])
        self._write_lines(a_tgt, ["X"])
        self._write_lines(k_src, ["y"])
        self._write_lines(k_tgt, ["Y"])

        out_dir = tmp_path / "new" / "nested"
        merge_datasets(a_src, a_tgt, k_src, k_tgt, out_dir)
        assert out_dir.exists()

    def test_output_filenames(self, tmp_path):
        a_src = tmp_path / "a.cs"
        a_tgt = tmp_path / "a.de"
        k_src = tmp_path / "k.cs"
        k_tgt = tmp_path / "k.de"

        self._write_lines(a_src, ["x"])
        self._write_lines(a_tgt, ["X"])
        self._write_lines(k_src, ["y"])
        self._write_lines(k_tgt, ["Y"])

        out_dir = tmp_path / "out"
        merged_src, merged_tgt = merge_datasets(a_src, a_tgt, k_src, k_tgt, out_dir)
        assert merged_src.name == "train_kd.cs"
        assert merged_tgt.name == "train_kd.de"

    def test_empty_kd(self, tmp_path):
        """Merging with empty KD data should give just authentic data."""
        a_src = tmp_path / "a.cs"
        a_tgt = tmp_path / "a.de"
        k_src = tmp_path / "k.cs"
        k_tgt = tmp_path / "k.de"

        self._write_lines(a_src, ["a1", "a2"])
        self._write_lines(a_tgt, ["A1", "A2"])
        # Empty KD files
        k_src.write_text("")
        k_tgt.write_text("")

        out_dir = tmp_path / "out"
        merged_src, _ = merge_datasets(a_src, a_tgt, k_src, k_tgt, out_dir)

        with open(merged_src) as f:
            lines = f.read().strip().splitlines()
        assert len(lines) == 2
