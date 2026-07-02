"""Tests for CLI helpers and the end-to-end entry point."""

from __future__ import annotations

import csv

import pytest

from picos_gc.cli import _collect_batches, _natural_key, main


def test_natural_key_orders_numerically(tmp_path):
    names = ["2pe10.gcd", "2pe2.gcd", "2pe1.gcd"]
    paths = [tmp_path / n for n in names]
    ordered = sorted(paths, key=_natural_key)
    assert [p.name for p in ordered] == ["2pe1.gcd", "2pe2.gcd", "2pe10.gcd"]


def test_collect_batches_loose_files_and_dirs(tmp_path):
    # Loose files.
    f1 = tmp_path / "a.gcd"
    f2 = tmp_path / "b.gcd"
    f1.touch()
    f2.touch()
    # A directory holding .gcd files directly.
    d = tmp_path / "run1"
    d.mkdir()
    (d / "x.gcd").touch()
    (d / "y.gcd").touch()

    batches = _collect_batches([f1, f2, d])
    labels = [label for label, _ in batches]
    assert "batch" in labels  # loose files grouped under "batch"
    assert "run1" in labels
    run1 = dict(batches)["run1"]
    assert {p.name for p in run1} == {"x.gcd", "y.gcd"}


def test_collect_batches_nested_one_level(tmp_path):
    parent = tmp_path / "parent"
    sub1 = parent / "sub1"
    sub2 = parent / "sub2"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)
    (sub1 / "a.gcd").touch()
    (sub2 / "b.gcd").touch()

    batches = _collect_batches([parent])
    labels = {label for label, _ in batches}
    assert labels == {"sub1", "sub2"}


def test_main_missing_path_exits(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "does_not_exist.gcd")])


def test_main_rejects_even_smooth_window(example_gcd, tmp_path):
    with pytest.raises(SystemExit):
        main([str(example_gcd), "--smooth-window", "10", "--outdir", str(tmp_path)])


def test_main_end_to_end_writes_csv(example_gcd, tmp_path):
    main([str(example_gcd), "--outdir", str(tmp_path), "--align-tol", "0"])
    csv_path = tmp_path / "batch" / "resultados_integracion.csv"
    assert csv_path.exists()
    rows = list(csv.reader(csv_path.open()))
    assert rows[0] == [
        "sample_name",
        "filename",
        "peak_n",
        "tR_min",
        "height_mV",
        "area_mV_min",
        "left_min",
        "right_min",
    ]
    # At least one detected-peak data row for the file (peak_n now at index 2).
    data = [r for r in rows[1:] if r and r[2] != ""]
    assert len(data) >= 1


def test_baseline_defaults_to_arpls(example_gcd, tmp_path, monkeypatch):
    import picos_gc.cli as cli

    captured = {}

    def fake_run_batch(
        label,
        filepaths,
        params,
        output_base,
        align_tol,
        plot,
        split_mode="valley",
        baseline_method=None,
        baseline_lam=None,
    ):
        captured["method"] = baseline_method
        captured["lam"] = baseline_lam

    monkeypatch.setattr(cli, "_run_batch", fake_run_batch)
    main([str(example_gcd), "--outdir", str(tmp_path)])
    assert captured["method"] == "arpls"
    assert captured["lam"] is None


def test_baseline_none_disables(example_gcd, tmp_path, monkeypatch):
    import picos_gc.cli as cli

    captured = {}

    def fake_run_batch(
        label,
        filepaths,
        params,
        output_base,
        align_tol,
        plot,
        split_mode="valley",
        baseline_method=None,
        baseline_lam=None,
    ):
        captured["method"] = baseline_method

    monkeypatch.setattr(cli, "_run_batch", fake_run_batch)
    main([str(example_gcd), "--baseline", "none", "--outdir", str(tmp_path)])
    assert captured["method"] is None


def test_cli_accepts_deconvolve_split_mode(example_gcd, tmp_path, monkeypatch):
    import picos_gc.cli as cli

    captured = {}

    def fake_run_batch(
        label,
        filepaths,
        params,
        output_base,
        align_tol,
        plot,
        split_mode="valley",
        baseline_method=None,
        baseline_lam=None,
    ):
        captured["split_mode"] = split_mode

    monkeypatch.setattr(cli, "_run_batch", fake_run_batch)
    main([str(example_gcd), "--split-mode", "deconvolve", "--outdir", str(tmp_path)])
    assert captured["split_mode"] == "deconvolve"
