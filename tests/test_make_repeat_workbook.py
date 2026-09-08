import pytest

pytest.importorskip("openpyxl")
import openpyxl

from make_repeat_workbook import SRC, trim


def test_trim_hides_everything_but_the_repeat_tubes():
    wb = openpyxl.load_workbook(SRC)
    trim(wb, {"C4", "BIN1"})
    lab = wb["Lab_DES"]
    visible = [lab[f"A{r}"].value for r in range(10, 67) if not lab.row_dimensions[r].hidden]
    assert visible == ["C4", "#", "BIN1"]  # "#" = the binary header row 58
    assert (lab["B25"].value, lab["D25"].value) == ("ThyCarvac", "Carvacrol")  # back-filled
    feed = wb["2-phenylethanol_DES"]
    assert not feed.row_dimensions[21].hidden  # C4's feed row (Lab_DES!G25 -> I21)
    assert all(feed.row_dimensions[r].hidden for r in (6, 20, 22))
    assert feed["B21"].value == "ThyCarvac"
    gc = wb["Sheet2"]
    assert [r for r in range(4, 43) if not gc.row_dimensions[r].hidden] == [4, 5, 6, 7]
    assert gc["D44"].value == "GC Vial ID"
    assert [gc[f"D{r}"].value for r in range(45, 49)] == ["C4-T1", "C4-T2", "C4-B1", "C4-B2"]
    assert (gc["A45"].value, gc["B47"].value, gc["C48"].value) == ("C4", "Bot", 2)
    assert gc["A49"].value is None  # nothing past the four vials of the single ternary tube
    # DES prep tables: only ThyCarvac (C4) and ThyCar (BIN1) remain, on both variants
    dd = wb["datos_des"]
    assert [dd[f"B{r}"].value for r in range(17, 26) if not dd.row_dimensions[r].hidden] == [
        "ThyCar",
        "ThyCarvac",
    ]
    s1 = wb["Sheet1"]  # row 4 = ThyCar | ThyEug, row 6 = ThyCarvac | CamEug
    assert [r for r in range(4, 13) if not s1.row_dimensions[r].hidden] == [4, 6]


def test_unknown_tube_is_an_error():
    with pytest.raises(ValueError, match="Z9"):
        trim(openpyxl.load_workbook(SRC), {"Z9"})
