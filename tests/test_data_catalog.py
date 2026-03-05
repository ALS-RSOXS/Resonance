from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from resonance.api.data.catalog import Catalog, LazyImageSequence
from resonance.api.data.models import SampleMetadata
from resonance.api.data.writer import RunWriter


def _write_test_db(path: Path, n_events: int = 3) -> Path:
    sample = SampleMetadata(name="PS", formula="C8H8")
    w = RunWriter(path, sample)
    w.open()
    w.open_run("nexafs")
    w.open_stream("primary", {"Energy": {"dtype": "number"}})
    for i in range(n_events):
        w.write_event({"Energy": 280.0 + i})
    w.close_run()
    w.close()
    return path


def test_catalog_recent(tmp_path: Path) -> None:
    path = _write_test_db(tmp_path / "test.db")
    with Catalog(path) as cat:
        results = cat.recent(10)
    assert len(results) == 1
    assert results[0].plan_name == "nexafs"
    assert results[0].sample_name == "PS"


def test_catalog_by_sample(tmp_path: Path) -> None:
    path = _write_test_db(tmp_path / "test.db")
    with Catalog(path) as cat:
        found = cat.by_sample("PS")
        missing = cat.by_sample("nonexistent")
    assert len(found) == 1
    assert missing == []


def test_catalog_getitem(tmp_path: Path) -> None:
    path = _write_test_db(tmp_path / "test.db")
    with Catalog(path) as cat:
        uid = cat.recent(1)[0].uid
        run = cat[uid]
        assert run.uid == uid


def test_catalog_getitem_missing_raises_keyerror(tmp_path: Path) -> None:
    path = _write_test_db(tmp_path / "test.db")
    with Catalog(path) as cat, pytest.raises(KeyError):
        cat["nonexistent_uid"]


def test_run_table(tmp_path: Path) -> None:
    path = _write_test_db(tmp_path / "test.db")
    with Catalog(path) as cat:
        run = cat[cat.recent(1)[0].uid]
        df = run.table("primary")
    assert len(df) == 3
    assert "seq_num" in df.columns
    assert "time" in df.columns
    assert "Energy" in df.columns
    assert list(df["Energy"]) == [280.0, 281.0, 282.0]


def test_run_sample_metadata(tmp_path: Path) -> None:
    path = _write_test_db(tmp_path / "test.db")
    with Catalog(path) as cat:
        run = cat[cat.recent(1)[0].uid]
        sample = run.sample
    assert sample is not None
    assert sample.name == "PS"
    assert sample.formula == "C8H8"


def test_run_table_empty_stream(tmp_path: Path) -> None:
    path = _write_test_db(tmp_path / "test.db")
    with Catalog(path) as cat:
        run = cat[cat.recent(1)[0].uid]
        df = run.table("nonexistent_stream")
    assert len(df) == 0
    assert "seq_num" in df.columns
    assert "time" in df.columns


def test_catalog_context_manager(tmp_path: Path) -> None:
    path = _write_test_db(tmp_path / "test.db")
    with Catalog(path) as cat:
        results = cat.recent()
    assert len(results) == 1


def _write_test_db_with_images(path: Path, n_images: int = 2) -> Path:
    sample = SampleMetadata(name="PS")
    w = RunWriter(path, sample)
    w.open()
    w.open_run("nexafs")
    w.open_stream("primary", {"detector_image": {"dtype": "int32", "external": True}})
    for i in range(n_images):
        event_uid = w.write_event({"i": float(i)})
        w.write_image(event_uid, "detector_image", np.ones((4, 4), dtype=np.int32) * i)
    w.close_run()
    w.close()
    return path


def test_run_images_shape(tmp_path: Path) -> None:
    path = _write_test_db_with_images(tmp_path / "bt.db", n_images=3)
    with Catalog(path) as cat:
        run = cat[cat.recent(1)[0].uid]
        imgs = run.images()
    assert len(imgs) == 3
    assert imgs.shape == (3, 4, 4)


def test_run_images_getitem_int(tmp_path: Path) -> None:
    path = _write_test_db_with_images(tmp_path / "bt.db", n_images=2)
    with Catalog(path) as cat:
        run = cat[cat.recent(1)[0].uid]
        imgs = run.images()
    np.testing.assert_array_equal(imgs[0], np.zeros((4, 4), dtype=np.int32))
    np.testing.assert_array_equal(imgs[1], np.ones((4, 4), dtype=np.int32))


def test_run_images_getitem_slice(tmp_path: Path) -> None:
    path = _write_test_db_with_images(tmp_path / "bt.db", n_images=3)
    with Catalog(path) as cat:
        run = cat[cat.recent(1)[0].uid]
        imgs = run.images()
    sliced = imgs[0:2]
    assert sliced.shape == (2, 4, 4)


def test_run_images_empty_field(tmp_path: Path) -> None:
    path = _write_test_db_with_images(tmp_path / "bt.db", n_images=2)
    with Catalog(path) as cat:
        run = cat[cat.recent(1)[0].uid]
        imgs = run.images("nonexistent_field")
    assert isinstance(imgs, LazyImageSequence)
    assert len(imgs) == 0
