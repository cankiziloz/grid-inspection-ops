import importlib.util
from pathlib import Path


def load_simulator():
    path = Path(__file__).parent.parent / "scripts" / "simulate_edge_feed.py"
    spec = importlib.util.spec_from_file_location("simulate_edge_feed", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_discover_images_can_filter_by_requested_case_folder(tmp_path):
    for case in ("glue", "thread", "bent"):
        case_dir = tmp_path / case
        case_dir.mkdir()
        (case_dir / f"{case}.png").write_bytes(b"fixture")

    simulator = load_simulator()

    selected = simulator.discover_images(tmp_path, {"glue", "thread"})

    assert {path.parent.name for path in selected} == {"glue", "thread"}
