import pytest
from reactome_tool import get_drugs_for_pathway

def test_valid_pathway_returns_drugs(monkeypatch):
    def mock_fetch_json(url):
        if "pathways" in url:
            return []  # no subpathways
        if "participants" in url:
            return [{"dbId": 123, "schemaClass": "ChemicalDrug", "displayName": "Aspirin [Reactome]", "referenceEntity": {"identifier": "ASP"}}]
        if "query/123" in url:
            return {"schemaClass": "ChemicalDrug", "displayName": "Aspirin [Reactome]", "referenceEntity": {"identifier": "ASP"}}
        return {}

    monkeypatch.setattr("reactome_tool.fetch_json", mock_fetch_json)
    drugs = get_drugs_for_pathway("R-HSA-109581")
    assert "Aspirin" in drugs
