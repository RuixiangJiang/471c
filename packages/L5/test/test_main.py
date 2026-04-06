from pathlib import Path

from click.testing import CliRunner
from L3.syntax import Program as L3Program

from L5 import main as l5_main_module


def test_main_pipeline_with_check_and_optimize(monkeypatch, tmp_path: Path):
    calls = []

    def fake_dummy_parse(code: str):
        calls.append(("dummy_parse", code))
        return "L5"

    def fake_convert_to_l3(program):
        calls.append(("convert_to_l3", program))
        return "L3"

    def fake_check_program(program):
        calls.append(("check_program", program))

    def fake_uniqify_program(program):
        calls.append(("uniqify_program", program))
        return ("fresh", "L3-uniq")

    def fake_eliminate_letrec_program(program):
        calls.append(("eliminate_letrec_program", program))
        return "L2"

    def fake_optimize_program(program):
        calls.append(("optimize_program", program))
        return "L2-opt"

    def fake_cps_convert_program(program, fresh):
        calls.append(("cps_convert_program", program, fresh))
        return "L1"

    def fake_to_ast_program(program):
        calls.append(("to_ast_program", program))
        return "PYTHON_CODE"

    monkeypatch.setattr(l5_main_module, "dummy_parse", fake_dummy_parse)
    monkeypatch.setattr(l5_main_module, "convert_to_l3", fake_convert_to_l3)
    monkeypatch.setattr(l5_main_module, "check_program", fake_check_program)
    monkeypatch.setattr(l5_main_module, "uniqify_program", fake_uniqify_program)
    monkeypatch.setattr(l5_main_module, "eliminate_letrec_program", fake_eliminate_letrec_program)
    monkeypatch.setattr(l5_main_module, "optimize_program", fake_optimize_program)
    monkeypatch.setattr(l5_main_module, "cps_convert_program", fake_cps_convert_program)
    monkeypatch.setattr(l5_main_module, "to_ast_program", fake_to_ast_program)

    input_file = tmp_path / "example.l5"
    input_file.write_text("source text")

    runner = CliRunner()
    result = runner.invoke(l5_main_module.main, [str(input_file)])

    assert result.exit_code == 0
    assert input_file.with_suffix(".py").read_text() == "PYTHON_CODE"

    assert calls == [
        ("dummy_parse", "source text"),
        ("convert_to_l3", "L5"),
        ("check_program", "L3"),
        ("uniqify_program", "L3"),
        ("eliminate_letrec_program", "L3-uniq"),
        ("optimize_program", "L2"),
        ("cps_convert_program", "L2-opt", "fresh"),
        ("to_ast_program", "L1"),
    ]


def test_main_pipeline_without_check_or_optimize_and_with_explicit_output(monkeypatch, tmp_path: Path):
    calls = []

    def fake_dummy_parse(code: str):
        calls.append(("dummy_parse", code))
        return "L5"

    def fake_convert_to_l3(program):
        calls.append(("convert_to_l3", program))
        return "L3"

    def fake_check_program(program):
        calls.append(("check_program", program))

    def fake_uniqify_program(program):
        calls.append(("uniqify_program", program))
        return ("fresh2", "L3-uniq2")

    def fake_eliminate_letrec_program(program):
        calls.append(("eliminate_letrec_program", program))
        return "L2-raw"

    def fake_optimize_program(program):
        calls.append(("optimize_program", program))
        return "L2-opt"

    def fake_cps_convert_program(program, fresh):
        calls.append(("cps_convert_program", program, fresh))
        return "L1"

    def fake_to_ast_program(program):
        calls.append(("to_ast_program", program))
        return "OUT"

    monkeypatch.setattr(l5_main_module, "dummy_parse", fake_dummy_parse)
    monkeypatch.setattr(l5_main_module, "convert_to_l3", fake_convert_to_l3)
    monkeypatch.setattr(l5_main_module, "check_program", fake_check_program)
    monkeypatch.setattr(l5_main_module, "uniqify_program", fake_uniqify_program)
    monkeypatch.setattr(l5_main_module, "eliminate_letrec_program", fake_eliminate_letrec_program)
    monkeypatch.setattr(l5_main_module, "optimize_program", fake_optimize_program)
    monkeypatch.setattr(l5_main_module, "cps_convert_program", fake_cps_convert_program)
    monkeypatch.setattr(l5_main_module, "to_ast_program", fake_to_ast_program)

    input_file = tmp_path / "input.l5"
    output_file = tmp_path / "result.py"
    input_file.write_text("abc")

    runner = CliRunner()
    result = runner.invoke(
        l5_main_module.main,
        ["--no-check", "--no-optimize", "-o", str(output_file), str(input_file)],
    )

    assert result.exit_code == 0
    assert output_file.read_text() == "OUT"

    assert calls == [
        ("dummy_parse", "abc"),
        ("convert_to_l3", "L5"),
        ("uniqify_program", "L3"),
        ("eliminate_letrec_program", "L3-uniq2"),
        ("cps_convert_program", "L2-raw", "fresh2"),
        ("to_ast_program", "L1"),
    ]