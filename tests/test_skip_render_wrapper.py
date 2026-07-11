from __future__ import annotations

import time
from pathlib import Path

from avagen.renderers.liveportrait_wrapper import (
    LivePortraitRunConfig,
    run_liveportrait_inference,
)

# Fake inference.py: write the driving template (<driving>.pkl) immediately,
# then simulate the long, memory-heavy render we want to skip by sleeping.
FAKE_INFERENCE = '''
import sys
import time
from pathlib import Path

argv = sys.argv
driving = argv[argv.index("-d") + 1]
Path(driving).with_suffix(".pkl").write_bytes(b"motion-template-bytes")
time.sleep(120)
'''


def _setup(tmp_path: Path) -> LivePortraitRunConfig:
    root = tmp_path / "LivePortrait"
    root.mkdir()
    (root / "inference.py").write_text(FAKE_INFERENCE, encoding="utf-8")
    source = tmp_path / "source.png"
    source.write_text("png", encoding="utf-8")
    driving = tmp_path / "driving.mp4"
    driving.write_text("mp4", encoding="utf-8")
    output_dir = tmp_path / "out"
    return LivePortraitRunConfig(
        source_path=source,
        driving_path=driving,
        output_dir=output_dir,
        liveportrait_root=root,
        stop_when_file=driving.with_suffix(".pkl"),
        stop_poll_interval=0.1,
        stop_stable_checks=2,
    )


def test_stops_once_template_is_written(tmp_path: Path) -> None:
    config = _setup(tmp_path)
    start = time.monotonic()
    result = run_liveportrait_inference(config)
    elapsed = time.monotonic() - start

    assert result.status == "completed"
    assert result.returncode == 0
    # The fake sleeps 120s; we must terminate it well before that.
    assert elapsed < 30
    assert config.driving_path.with_suffix(".pkl").read_bytes() == b"motion-template-bytes"
