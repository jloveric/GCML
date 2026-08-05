from pathlib import Path
import py_compile
import sys

import hydra
import matplotlib.pyplot as plt


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from scripts.plotting import FigureWriter  # noqa: E402

EXPERIMENTS = ("gcml_abstract_graph", "gcml_grid_cell", "gcml_tiling")


def test_notebooks_were_replaced() -> None:
    assert not list(ROOT.glob("*.ipynb"))


def test_experiment_scripts_compile() -> None:
    for experiment in EXPERIMENTS:
        py_compile.compile(ROOT / "scripts" / f"{experiment}.py", doraise=True)


def test_hydra_configs_compose() -> None:
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        for experiment in EXPERIMENTS:
            config = hydra.compose(config_name=experiment)
            assert "output_dir" in config


def test_scripts_do_not_contain_machine_specific_paths() -> None:
    for script in (ROOT / "scripts").glob("*.py"):
        assert "/home/linhui" not in script.read_text()


def test_scripts_do_not_display_figures_interactively() -> None:
    for script in (ROOT / "scripts").glob("gcml_*.py"):
        assert "plt.show()" not in script.read_text()


def test_experiment_scripts_do_not_reference_undeclared_notebook_state() -> None:
    grid_script = (ROOT / "scripts" / "gcml_grid_cell.py").read_text()
    assert "probs.max()" not in grid_script

    tiling_script = (ROOT / "scripts" / "gcml_tiling.py").read_text()
    assert 'db_dir="./dataset/' not in tiling_script


def test_configs_use_timestamped_hydra_output() -> None:
    for experiment in EXPERIMENTS:
        text = (ROOT / "configs" / f"{experiment}.yaml").read_text()
        assert "${hydra:runtime.output_dir}" in text
        assert "${now:%Y-%m-%d}" in text
        assert "${now:%H-%M-%S}" in text


def test_figure_writer_saves_and_closes_figures(tmp_path: Path) -> None:
    plt.figure()
    plt.plot([0, 1], [0, 1])

    paths = FigureWriter(tmp_path, "test", dpi=72).save_all()

    assert paths == [tmp_path / "images" / "test_001.png"]
    assert paths[0].is_file()
    assert not plt.get_fignums()
