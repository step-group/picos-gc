import pytest

pytest.importorskip("openpyxl")
import openpyxl

from make_repeat_workbook import SRC, trim


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
    # 10 g DES prep table: only the three DES in use remain; the 5 g sheet is hidden
    dd = wb["datos_des"]
    assert [dd[f"B{r}"].value for r in range(17, 26) if not dd.row_dimensions[r].hidden] == [
        "ThyCar",
        "ThyCarvac",
        "CamEug",
    ]
    assert (dd["G17"].value, dd["L17"].value) == (None, None)  # last campaign's weighings gone
    assert str(dd["M17"].value).startswith("=IF(")  # the ratio formula stays
    assert wb["Sheet1"].sheet_state == "hidden"
    assert wb["Sheet2"].sheet_state == "hidden"  # superseded by Sampling
    # Sampling (from PLANTILLA_IMPRESION.xlsx): block C header + C4 vials, binaries header + BIN1
    sp = wb["Sampling"]
    visible = [r for r in range(1, 220) if not sp.row_dimensions[r].hidden]
    assert visible == [47, 48, 49, 62, 63, 64, 65, 185, 186, 187, 188, 189, 190, 191]
    assert (sp["D62"].value, sp["D188"].value) == ("C4-T1", "BIN1-T1")
    heights = sp.row_dimensions  # binaries rows re-heighted like the ternary block
    assert (heights[187].height, heights[188].height) == (heights[3].height, heights[4].height)
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
