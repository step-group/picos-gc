import pytest

pytest.importorskip("openpyxl")
import openpyxl

from make_repeat_workbook import SAMPLING_HEAD_H, SAMPLING_ROW_H, SRC, trim


def test_trim_hides_everything_but_the_repeat_tubes():
    wb = openpyxl.load_workbook(SRC)
    trim(wb, {"C4", "BIN1", "I1"})
    lab = wb["Lab_DES"]
    visible = [lab[f"A{r}"].value for r in range(10, 67) if not lab.row_dimensions[r].hidden]
    assert visible == ["C4", "I1", "#", "BIN1"]  # "#" = the binary header row 58
    assert (lab["B25"].value, lab["D25"].value) == ("ThyCarvac", "Carvacrol")  # back-filled
    # only the volume guide stays: weighing/record columns hidden, their labels cleared
    assert all(lab.column_dimensions[c].hidden for c in "JKLMNOPQRST")
    assert not lab.column_dimensions["I"].hidden and not lab.column_dimensions["U"].hidden
    assert (lab["B7"].value, lab["D7"].value, lab["F7"].value) == ("Guía", None, None)
    feed = wb["2-phenylethanol_DES"]
    assert not feed.row_dimensions[21].hidden  # C4's feed row (Lab_DES!G25 -> I21)
    assert all(feed.row_dimensions[r].hidden for r in (6, 20, 22))
    assert feed["B21"].value == "ThyCarvac"
    # 10 g DES prep table: only the DES the TERNARY tubes use stay in the table itself,
    # and BIN1's ThyCar is not one of them -- it gets its own Falcon row below.
    dd = wb["datos_des"]
    assert [dd[f"B{r}"].value for r in range(17, 26) if not dd.row_dimensions[r].hidden] == [
        "ThyCarvac",
        "CamEug",
    ]
    assert (dd["G17"].value, dd["L17"].value) == (None, None)  # last campaign's weighings gone
    assert str(dd["M17"].value).startswith("=IF(")  # the ratio formula stays
    assert dd["B29"].value == "Binarios: Falcon aparte"
    assert dd["B30"].value == "ThyCar (BIN1)"  # duplicated from row 17, formulas re-pointed
    assert (dd["C30"].value, dd["H30"].value) == ("thymol", "carvone")
    assert dd["D30"].value == "=VLOOKUP(C30,Componentes!$AA$3:$AC$17,2,FALSE())"
    assert dd["F30"].value == "=$C$15*D30/(D30+I30)/E30"  # $-anchored refs untouched
    assert (dd["G30"].value, dd["L30"].value) == (None, None)
    assert dd["B31"].value is None  # one binary in, one row out
    assert dd.row_dimensions[2].hidden and not dd.row_dimensions[14].hidden  # status matrix gone
    assert wb["Sheet1"].sheet_state == "hidden"
    assert wb["Sheet2"].sheet_state == "hidden"  # superseded by Sampling
    # Sampling (from PLANTILLA_IMPRESION.xlsx): block C header + C4 vials, binaries header + BIN1
    sp = wb["Sampling"]
    visible = [r for r in range(1, 220) if not sp.row_dimensions[r].hidden]
    assert visible == [47, 48, 49, 62, 63, 64, 65, 185, 186, 187, 188, 189, 190, 191]
    assert (sp["D62"].value, sp["D188"].value) == ("C4-T1", "BIN1-T1")
    # ternary vials: 300 µL sample + 700 µL IPA (was 200 + 800), method line's DF follows
    assert (sp["I47"].value, sp["K47"].value) == ("300", "+700 µL IPA")
    assert "DF≈3.3" in sp["A48"].value and "DF=5" not in sp["A48"].value
    assert (sp["I221"].value, sp["K221"].value) == ("300", "+700 µL IPA")  # cloned block too
    assert (sp["I185"].value, sp["K185"].value) == (300, "+700 µL IPA")  # binaries untouched
    # one writable height everywhere: ternary (15 pt), binary (70.85 pt) and cloned rows
    assert {sp.row_dimensions[r].height for r in (4, 62, 188, 224)} == {SAMPLING_ROW_H}
    assert {sp.row_dimensions[r].height for r in (3, 187, 223)} == {SAMPLING_HEAD_H}
    assert sp["H62"].value == '=IF(OR(E62="",F62=""),"",F62-E62)'  # formulas survived the copy
    assert sp["A47"].font.b and sp["E62"].fill.fgColor.rgb == "FFFCE4D6"  # and so did styles
    # block I is not in the template: cloned after the binaries, titled from Lab_DES
    assert sp["A221"].value == "I — CamEug  (Camphor + Eugenol)"
    assert sp["A223"].value == "System"
    assert [sp[f"D{r}"].value for r in range(224, 228)] == ["I1-T1", "I1-T2", "I1-B1", "I1-B2"]
    assert (sp["B224"].value, sp["B226"].value, sp["C227"].value) == ("Top", "Bot", 2)
    assert sp["J227"].value == '=IF(OR(H227="",I227=""),"",(H227+I227)/H227)'
    assert sp["A228"].value is None


def test_unknown_tube_is_an_error():
    with pytest.raises(ValueError, match="Z9"):
        trim(openpyxl.load_workbook(SRC), {"Z9"})
