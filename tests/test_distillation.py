"""Tests for src/distillation/ -- merge_datasets and generate_kd_data."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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


class TestGenerateKdData:
    """Tests for generate_kd_data, with vLLM and COMET mocked."""

    def _make_src_ref(self, tmp_path, src_lines, ref_lines):
        src = tmp_path / "train.cs"
        ref = tmp_path / "train.de"
        src.write_text("\n".join(src_lines) + "\n")
        ref.write_text("\n".join(ref_lines) + "\n")
        return src, ref

    @patch("src.distillation.generate_kd.translate_with_vllm")
    @patch("src.distillation.generate_kd.compute_comet")
    def test_newlines_stripped_from_hypotheses(self, mock_comet, mock_vllm, tmp_path):
        """Teacher outputs with embedded newlines must be flattened to spaces."""
        from src.distillation.generate_kd import generate_kd_data

        sources = ["Ahoj", "Dobry den"]
        refs = ["Hallo", "Guten Tag"]
        src_path, ref_path = self._make_src_ref(tmp_path, sources, refs)
        out_dir = tmp_path / "kd"

        # Teacher returns hypotheses with embedded newlines
        mock_vllm.return_value = ["Hallo\nWelt", "Guten\nTag\nhier"]

        # Mock COMET: patch the download_model and load_from_checkpoint
        mock_comet_model = MagicMock()
        mock_comet_model.predict.return_value = MagicMock(scores=[0.9, 0.8])
        with patch("comet.download_model", return_value="/fake"), \
             patch("comet.load_from_checkpoint", return_value=mock_comet_model):
            generate_kd_data(
                src_path=src_path, ref_path=ref_path, output_dir=out_dir,
                comet_threshold=0.7,
            )

        kd_src = (out_dir / "kd.cs").read_text().strip().splitlines()
        kd_tgt = (out_dir / "kd.de").read_text().strip().splitlines()

        assert len(kd_src) == len(kd_tgt) == 2
        assert "\n" not in kd_tgt[0]
        assert kd_tgt[0] == "Hallo Welt"
        assert kd_tgt[1] == "Guten Tag hier"

    @patch("src.distillation.generate_kd.translate_with_vllm")
    def test_custom_extensions(self, mock_vllm, tmp_path):
        """Output files should use src_ext/tgt_ext, not hardcoded cs/de."""
        from src.distillation.generate_kd import generate_kd_data

        src_path = tmp_path / "train.en"
        ref_path = tmp_path / "train.es"
        src_path.write_text("Hello\n")
        ref_path.write_text("Hola\n")
        out_dir = tmp_path / "kd"

        mock_vllm.return_value = ["Hola"]
        mock_comet_model = MagicMock()
        mock_comet_model.predict.return_value = MagicMock(scores=[0.95])
        with patch("comet.download_model", return_value="/fake"), \
             patch("comet.load_from_checkpoint", return_value=mock_comet_model):
            generate_kd_data(
                src_path=src_path, ref_path=ref_path, output_dir=out_dir,
                comet_threshold=0.7, src_ext="en", tgt_ext="es",
            )

        assert (out_dir / "kd.en").exists()
        assert (out_dir / "kd.es").exists()
        assert not (out_dir / "kd.cs").exists()

    @patch("src.distillation.generate_kd.translate_with_vllm")
    def test_comet_filtering(self, mock_vllm, tmp_path):
        """Pairs below COMET threshold should be excluded."""
        from src.distillation.generate_kd import generate_kd_data

        sources = ["s1", "s2", "s3"]
        refs = ["r1", "r2", "r3"]
        src_path, ref_path = self._make_src_ref(tmp_path, sources, refs)
        out_dir = tmp_path / "kd"

        mock_vllm.return_value = ["h1", "h2", "h3"]
        mock_comet_model = MagicMock()
        # Only first and third pass the threshold
        mock_comet_model.predict.return_value = MagicMock(scores=[0.9, 0.5, 0.8])
        with patch("comet.download_model", return_value="/fake"), \
             patch("comet.load_from_checkpoint", return_value=mock_comet_model):
            generate_kd_data(
                src_path=src_path, ref_path=ref_path, output_dir=out_dir,
                comet_threshold=0.7,
            )

        kd_src = (out_dir / "kd.cs").read_text().strip().splitlines()
        kd_tgt = (out_dir / "kd.de").read_text().strip().splitlines()
        assert len(kd_src) == len(kd_tgt) == 2
        assert kd_src == ["s1", "s3"]
        assert kd_tgt == ["h1", "h3"]
