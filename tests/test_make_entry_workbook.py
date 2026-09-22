import pytest

pytest.importorskip("openpyxl")

from make_entry_workbook import FIRST, build, carry, typed


def test_kf_cells_only_on_organic_phases_fresh_or_reused():
    ws = build({"C4", "D1", "D3", "BIN1", "2PE"}).active
    row = {ws[f"B{r}"].value: r for r in range(FIRST, ws.max_row + 1)}
    assert len(row) == 20  # four vials per tube

    def kf(code):
        cell = ws[f"M{row[code]}"]
        return cell.value, cell.protection.locked

    assert kf("C4-T1") == (None, False)  # C4's organic is on top (campaign-1 KF 3.5 %)
    assert kf("D1-B1") == (None, False)  # re-measured: fresh, not the old 7.73
    assert kf("D3-B1") == (4.1294, True)  # reused campaign-1 KF, locked
    assert kf("D1-T1") == (None, True)  # aqueous: by difference, no KF
    assert kf("BIN1-T1") == (1.5599, True) and kf("BIN1-B1") == (None, True)  # reused
    assert kf("2PE-B1") == (None, False) and kf("2PE-T1") == (None, True)
    # masses are input everywhere, aqueous rows included
    assert not ws[f"E{row['D1-T1']}"].protection.locked
    assert ws.protection.sheet


def test_rebuild_keeps_what_was_typed_and_reports_what_it_cannot():
    old = build({"C4", "D3"}).active
    old["E9"], old["P9"], old["M9"] = 2.768, "cap cracked", 3.61  # C4-T1: mass, note, fresh KF
    inputs = typed(old)
    assert inputs == {"C4-T1": {"E": 2.768, "P": "cap cracked", "M": 3.61}}  # 300 µL default skipped
    new = build({"C4", "D2", "D3"}).active  # D2 lands between C4 and D3: rows shift
    assert carry(new, inputs) == []
    row = {new[f"B{r}"].value: r for r in range(FIRST, new.max_row + 1)}
    assert new[f"E{row['C4-T1']}"].value == 2.768 and new[f"M{row['C4-T1']}"].value == 3.61
    # a typed value whose cell went away (tube dropped, or KF cell now locked) is reported
    assert carry(build({"D3"}).active, inputs) == ["C4-T1 E=2.768", "C4-T1 M=3.61", "C4-T1 P=cap cracked"]
    assert carry(build({"C5"}).active, {"C5-T1": {"M": 5.7}}) == ["C5-T1 M=5.7"]  # aqueous: locked
