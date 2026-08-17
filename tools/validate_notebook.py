"""Static and offline smoke checks for the workshop notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "hands_on_agentes_jpr2026.ipynb"
REQUIREMENTS = ROOT / "requirements.txt"


def cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def select_code_cell(cells: list[dict], marker: str) -> str:
    matches = [cell_source(cell) for cell in cells if cell.get("cell_type") == "code" and marker in cell_source(cell)]
    if len(matches) != 1:
        raise AssertionError(f"Expected one code cell containing {marker!r}, found {len(matches)}")
    return matches[0]


def execute_selected_nodes(source: str, names: set[str], keep_imports: bool = False) -> dict:
    tree = ast.parse(source)
    selected: list[ast.stmt] = []
    for node in tree.body:
        if keep_imports and isinstance(node, (ast.Import, ast.ImportFrom)):
            selected.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            elif isinstance(node.target, ast.Name):
                targets = [node.target.id]
            if names.intersection(targets):
                selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {}
    exec(compile(module, "<selected-notebook-cell>", "exec"), namespace)
    return namespace


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert len(cells) == 38
    cell_ids = [cell.get("id") for cell in cells]
    assert all(cell_ids)
    assert len(set(cell_ids)) == len(cell_ids)

    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    compile_errors = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = cell_source(cell)
        try:
            compile(source, f"cell-{index}", "exec")
        except SyntaxError as exc:
            compile_errors.append(f"cell {index}: {exc}")
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
    assert not compile_errors, "\n".join(compile_errors)

    required_pins = {
        "agno==2.9.0",
        "google-genai==2.18.1",
        "requests==2.32.5",
    }
    requirement_lines = {
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert requirement_lines == required_pins
    setup_source = select_code_cell(cells, "PACKAGES = [")
    for pin in required_pins:
        assert f'"{pin}"' in setup_source
    assert 'GOOGLE_API_KEY' in setup_source
    assert 'gemini-3.6-flash' in setup_source

    worklist_source = select_code_cell(cells, "DEMO_CASES = {")
    worklist = execute_selected_nodes(
        worklist_source,
        {"DEMO_CASES", "lookup_demo_case"},
    )
    found = worklist["lookup_demo_case"]("rx-demo-004")
    missing = worklist["lookup_demo_case"]("RX-DEMO-999")
    assert found["found"] is True and found["source_id"] == "SYNTHETIC_WORKLIST_V1"
    assert missing == {
        "found": False,
        "accession": "RX-DEMO-999",
        "source_id": "SYNTHETIC_WORKLIST_V1",
    }

    claim_source = select_code_cell(cells, "CLAIM_2024_SNIPPETS = [")
    claim = execute_selected_nodes(
        claim_source,
        {"CLAIM_2024_SNIPPETS", "_tokens", "search_claim_2024"},
        keep_imports=True,
    )
    retrieved = claim["search_claim_2024"]("patient split leakage external validation", 4)
    retrieved_ids = {item["id"] for item in retrieved}
    assert "CLAIM-DATA-SPLIT" in retrieved_ids
    assert "CLAIM-EXTERNAL-VALIDATION" in retrieved_ids
    assert all(item["doi"] == "10.1148/ryai.240300" for item in retrieved)

    from agno.agent import Agent
    from agno.models.google import Gemini

    model = Gemini(
        id="gemini-3.6-flash",
        max_output_tokens=1400,
        thinking_level="low",
        timeout=90,
        retries=1,
    )
    agent = Agent(
        model=model,
        tools=[worklist["lookup_demo_case"]],
        tool_call_limit=1,
    )
    assert agent.tool_call_limit == 1
    assert model.id == "gemini-3.6-flash"
    assert model.max_output_tokens == 1400

    image_source = select_code_cell(cells, 'PUBLIC_IMAGES = {')
    assert 'thoracic_spine_xray.png' in image_source
    assert 'SPINE_XRAY_PATH' in image_source
    assert 'not_a_cxr' not in image_source

    router_source = select_code_cell(cells, 'BIOMEDCLIP_ID =')
    assert 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224' in router_source
    assert 'a full-length standing spine radiograph' in router_source
    assert 'scores are not calibrated probabilities' in router_source

    gate_source = select_code_cell(cells, 'def gated_vision_response')
    refusal_index = gate_source.index('if not route["accepted"]')
    model_call_index = gate_source.index('return show_response(')
    assert refusal_index < model_call_index
    assert 'SPINE_XRAY_PATH' in gate_source

    markdown = "\n".join(cell_source(cell) for cell in cells if cell.get("cell_type") == "markdown")
    all_source = "\n".join(cell_source(cell) for cell in cells)
    assert 'not_a_cxr' not in all_source
    assert 'Cat photograph' not in all_source
    for required_text in (
        "60-minute",
        "Trainee Editorial Board",
        "prepaid credits",
        "Google AI Studio",
        "BiomedCLIP",
        "CLAIM 2024",
        "CARE-X",
    ):
        assert required_text in markdown

    print(
        json.dumps(
            {
                "cells": len(cells),
                "code_cells": len(code_cells),
                "compile_errors": 0,
                "offline_tool_contracts": "pass",
                "agno_contracts": "pass",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
