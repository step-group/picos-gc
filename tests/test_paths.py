from paths import ENTRY, GC, OUT, ROOT, TEMPLATES, WORKBOOKS


def test_locations_are_absolute_and_rooted_at_the_repo():
    assert ROOT.is_absolute() and (ROOT / "pyproject.toml").exists()
    assert all(p.is_relative_to(ROOT) for p in (GC, WORKBOOKS, ENTRY, TEMPLATES, OUT))


def test_tracked_inputs_are_where_paths_says():
    assert (GC / "BINARIOS_TERPENOS").is_dir()
    for name in ("Sistemas ternarios_MF.xlsx", "CC_MF.xlsx", "aromas_equilibrios_vfinal.xlsx"):
        assert (WORKBOOKS / name).exists(), name
    assert (TEMPLATES / "PLANTILLA_IMPRESION.xlsx").exists()
    assert sorted(p.name for p in ENTRY.glob("repeat*_entry.xlsx")) == [
        "repeat2_entry.xlsx",
        "repeat3_entry.xlsx",
        "repeat4_entry.xlsx",
        "repeat_entry.xlsx",
    ]
