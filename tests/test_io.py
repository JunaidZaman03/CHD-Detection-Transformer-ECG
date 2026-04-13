from __future__ import annotations

from pathlib import Path

from chddecg.data.io import parse_header


def test_parse_header_extracts_core_fields(tmp_path: Path):
    hea = tmp_path / "E00001.hea"
    hea.write_text(
        "E00001 12 500 5000\n"
        "#Dx: 426783006\n"
        "#Age: 45\n"
        "#Sex: Female\n"
        "#Heart rate: 70\n",
        encoding="utf-8",
    )

    meta = parse_header(hea)
    assert meta.file_id == "E00001"
    assert meta.fs == 500
    assert meta.sex == "Female"
