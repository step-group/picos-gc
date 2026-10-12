from paths import ENTRY, GC, OUT, ROOT, TEMPLATES, WORKBOOKS


def test_locations_are_absolute_and_rooted_at_the_repo():
    assert ROOT.is_absolute() and (ROOT / "pyproject.toml").exists()
    assert all(p.is_relative_to(ROOT) for p in (GC, WORKBOOKS, ENTRY, TEMPLATES, OUT))


def test_tracked_inputs_are_where_paths_says():
    assert (GC / "BINARIOS_TERPENOS").is_dir()
    for name in ("Sistemas ternarios_MF.xlsx", "CC_MF.xlsx", "aromas_equilibrios_vfinal.xlsx"):
        assert (WORKBOOKS / name).exists(), name
    assert (TEMPLATES / "PLANTILLA_IMPRESION.xlsx").exists()


def test_every_round_reads_its_entry_sheet_from_data_entry():
    # a new round's sheet may sit in data/entry/ before it joins ROUNDS
    from repeat_results import ROUNDS

    sheets = [s for _, entry in ROUNDS for s in entry]
    assert sheets and all(s.parent == ENTRY and s.exists() for s in sheets), sheets
