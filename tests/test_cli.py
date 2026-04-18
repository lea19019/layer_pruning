"""Tests for the attention-lp CLI (src/cli.py).

Focus is on flag parsing and the flag-to-config-dict conversion. The actual
pipeline (model loading, pruning, fine-tuning) is NOT exercised here -- those
stages need real models and GPUs and are covered by their own unit tests.
"""

import pytest
import yaml

from src.cli import build_config_from_args, build_parser


def _parse_run(*argv: str):
    parser = build_parser()
    return parser.parse_args(["run", *argv])


class TestRunFlagParsing:
    def test_requires_exp_id_without_config(self):
        args = _parse_run()
        with pytest.raises(SystemExit):
            build_config_from_args(args)

    def test_minimal_flags(self):
        args = _parse_run("--exp-id", "foo")
        cfg = build_config_from_args(args)
        assert cfg["experiment_id"] == "foo"
        assert cfg["pruning"] == {"method": "none"}
        assert cfg["finetuning"] == {"enabled": False}
        assert cfg["distillation"] == {"enabled": False}
        assert cfg["quantization"] == {"enabled": False}

    def test_ifr_pruning_from_flags(self):
        args = _parse_run(
            "--exp-id", "i1",
            "--pruning", "ifr",
            "--n-remove", "8",
            "--scores-path", "scores.json",
        )
        cfg = build_config_from_args(args)
        assert cfg["pruning"]["method"] == "ifr"
        assert cfg["pruning"]["n_remove"] == 8
        assert cfg["pruning"]["scores_path"] == "scores.json"

    def test_heuristic_pruning_from_flags(self):
        args = _parse_run(
            "--exp-id", "m1",
            "--pruning", "heuristic",
            "--target-layers", "24",
            "--val-size", "100",
        )
        cfg = build_config_from_args(args)
        assert cfg["pruning"]["method"] == "heuristic"
        assert cfg["pruning"]["target_layers"] == 24
        assert cfg["pruning"]["val_size"] == 100

    def test_layers_to_remove_parses_comma_list(self):
        args = _parse_run(
            "--exp-id", "r1",
            "--pruning", "ifr",
            "--layers-to-remove", "3,5,7,11",
        )
        cfg = build_config_from_args(args)
        assert cfg["pruning"]["layers_to_remove"] == [3, 5, 7, 11]

    def test_finetune_flags(self):
        args = _parse_run(
            "--exp-id", "f1",
            "--finetune", "--qlora", "--epochs", "5",
        )
        cfg = build_config_from_args(args)
        assert cfg["finetuning"]["enabled"] is True
        assert cfg["finetuning"]["qlora"] is True
        assert cfg["finetuning"]["epochs"] == 5

    def test_kd_flag(self):
        args = _parse_run("--exp-id", "kd1", "--kd")
        cfg = build_config_from_args(args)
        assert cfg["distillation"]["enabled"] is True

    def test_quantize_flags(self):
        args = _parse_run(
            "--exp-id", "q1",
            "--quantize", "--bits", "4", "--quant-type", "nf4",
        )
        cfg = build_config_from_args(args)
        assert cfg["quantization"]["enabled"] is True
        assert cfg["quantization"]["bits"] == 4
        assert cfg["quantization"]["quant_type"] == "nf4"

    def test_lang_pair_passthrough(self):
        args = _parse_run("--exp-id", "l1", "--lang-pair", "en-es")
        cfg = build_config_from_args(args)
        assert cfg["lang_pair"] == "en-es"

    def test_data_and_output_dir_flags(self, tmp_path):
        data = tmp_path / "my_data"
        out = tmp_path / "my_out"
        args = _parse_run(
            "--exp-id", "d1",
            "--data-dir", str(data),
            "--output-dir", str(out),
            "--kd-dir", str(tmp_path / "my_kd"),
        )
        cfg = build_config_from_args(args)
        assert cfg["data_dir"] == str(data)
        assert cfg["output_dir"] == str(out)
        assert cfg["kd_dir"] == str(tmp_path / "my_kd")


class TestYAMLConfigAndOverride:
    def test_loads_yaml_when_no_flags(self, tmp_path):
        yaml_cfg = {
            "experiment_id": "from_yaml",
            "base_model": "m",
            "seed": 7,
            "lang_pair": "cs-de",
            "pruning": {"method": "ifr", "n_remove": 4},
            "finetuning": {"enabled": True, "epochs": 2, "qlora": True},
            "distillation": {"enabled": False},
            "quantization": {"enabled": False},
        }
        path = tmp_path / "c.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(yaml_cfg, f)

        args = _parse_run("--config", str(path))
        cfg = build_config_from_args(args)
        assert cfg["experiment_id"] == "from_yaml"
        assert cfg["pruning"]["n_remove"] == 4
        assert cfg["finetuning"]["qlora"] is True

    def test_flags_override_yaml(self, tmp_path):
        yaml_cfg = {
            "experiment_id": "base",
            "lang_pair": "cs-de",
            "pruning": {"method": "ifr", "n_remove": 4},
            "finetuning": {"enabled": False},
            "distillation": {"enabled": False},
            "quantization": {"enabled": False},
        }
        path = tmp_path / "c.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(yaml_cfg, f)

        args = _parse_run(
            "--config", str(path),
            "--lang-pair", "en-es",
            "--n-remove", "12",
            "--finetune",
        )
        cfg = build_config_from_args(args)
        assert cfg["lang_pair"] == "en-es"
        assert cfg["pruning"]["n_remove"] == 12
        assert cfg["pruning"]["method"] == "ifr"  # preserved from YAML
        assert cfg["finetuning"]["enabled"] is True

    def test_no_finetune_flag_overrides_yaml_true(self, tmp_path):
        yaml_cfg = {
            "experiment_id": "base",
            "pruning": {"method": "none"},
            "finetuning": {"enabled": True, "epochs": 3},
            "distillation": {"enabled": False},
            "quantization": {"enabled": False},
        }
        path = tmp_path / "c.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(yaml_cfg, f)

        args = _parse_run("--config", str(path), "--no-finetune")
        cfg = build_config_from_args(args)
        assert cfg["finetuning"]["enabled"] is False
        # epochs preserved even though FT disabled
        assert cfg["finetuning"]["epochs"] == 3


class TestSubcommandRegistration:
    def test_all_subcommands_registered(self):
        import argparse

        parser = build_parser()
        subparsers_action = next(
            a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
        )
        assert set(subparsers_action.choices) >= {"run", "score-ifr", "evaluate", "aggregate"}

    def test_unknown_subcommand_errors(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["bogus"])
