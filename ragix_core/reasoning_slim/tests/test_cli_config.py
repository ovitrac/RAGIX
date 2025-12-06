import json
from argparse import Namespace

from ContractiveReasoner import _apply_config_overrides, build_arg_parser


def test_config_overrides_defaults(tmp_path):
    cfg = {
        "model": "fake-model",
        "max_depth": 7,
        "print_events": True,
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    parser = build_arg_parser()
    defaults = vars(parser.parse_args([]))

    # CLI provides only question; config should override defaults
    args = parser.parse_args(["--config", str(cfg_path), "hello"])
    merged = _apply_config_overrides(args, defaults)

    assert merged.model == "fake-model"
    assert merged.max_depth == 7
    assert merged.print_events is True
