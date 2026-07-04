from __future__ import annotations

from pathlib import Path

from avagen.renderers.liveportrait_wrapper import (
    LivePortraitRunConfig,
    build_liveportrait_command,
    resolve_inference_script,
    run_liveportrait_inference,
)


def _write_fake_inference_script(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import argparse",
                "from pathlib import Path",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('-s')",
                "parser.add_argument('-d')",
                "parser.add_argument('-o')",
                "args = parser.parse_args()",
                "output_dir = Path(args.o)",
                "output_dir.mkdir(parents=True, exist_ok=True)",
                "(output_dir / 'wrapper_ok.txt').write_text(f'{args.s}\\n{args.d}\\n', encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )


def test_resolve_inference_script_from_root(tmp_path: Path) -> None:
    liveportrait_root = tmp_path / "LivePortrait"
    liveportrait_root.mkdir()
    inference_path = liveportrait_root / "inference.py"
    inference_path.write_text("print('ok')\n", encoding="utf-8")

    config = LivePortraitRunConfig(
        source_path=tmp_path / "source.png",
        driving_path=tmp_path / "driving.mp4",
        output_dir=tmp_path / "output",
        liveportrait_root=liveportrait_root,
    )

    assert resolve_inference_script(config) == inference_path.resolve()


def test_build_liveportrait_command_uses_expected_paths(tmp_path: Path) -> None:
    liveportrait_root = tmp_path / "LivePortrait"
    liveportrait_root.mkdir()
    inference_path = liveportrait_root / "inference.py"
    inference_path.write_text("print('ok')\n", encoding="utf-8")
    source_path = tmp_path / "source.png"
    driving_path = tmp_path / "driving.mp4"
    source_path.write_text("", encoding="utf-8")
    driving_path.write_text("", encoding="utf-8")

    config = LivePortraitRunConfig(
        source_path=source_path,
        driving_path=driving_path,
        output_dir=tmp_path / "output",
        liveportrait_root=liveportrait_root,
        python_executable="python3",
    )

    command = build_liveportrait_command(config)

    assert command[:2] == ["python3", str(inference_path.resolve())]
    assert command[2:] == [
        "-s",
        str(source_path.resolve()),
        "-d",
        str(driving_path.resolve()),
        "-o",
        str((tmp_path / "output").resolve()),
    ]


def test_run_liveportrait_inference_executes_external_script(tmp_path: Path) -> None:
    liveportrait_root = tmp_path / "LivePortrait"
    liveportrait_root.mkdir()
    inference_path = liveportrait_root / "inference.py"
    _write_fake_inference_script(inference_path)

    source_path = tmp_path / "source.png"
    driving_path = tmp_path / "driving.mp4"
    source_path.write_text("source", encoding="utf-8")
    driving_path.write_text("driving", encoding="utf-8")
    output_dir = tmp_path / "output"

    config = LivePortraitRunConfig(
        source_path=source_path,
        driving_path=driving_path,
        output_dir=output_dir,
        liveportrait_root=liveportrait_root,
        python_executable="python3",
    )

    result = run_liveportrait_inference(config)

    assert result.status == "completed"
    assert result.returncode == 0
    assert (output_dir / "wrapper_ok.txt").exists()
