"""
Unit tests for AFL Betting Model API.
"""
import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Force load of models by removing cached modules if needed
import api as api_module

client = TestClient(api_module.app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["models_loaded"], bool)
    # Models should be loaded since files exist
    assert data["models_loaded"] is True

def test_predict_valid():
    payload = {
        "year": 2026,
        "round": 5,
        "player": "Marcus Bontempelli",
        "team": "Western Bulldogs",
        "opponent": "Collingwood",
        "venue": "M.C.G.",
        "is_home": 1,
        "had_bye_last_week": 0,
        "opponent_avg_disposals": 23.5,
        "opponent_defensive_avg": 15.2,
        "venue_avg_disposals": 22.1,
        "disposals_last_5": 28.3,
        "disposals_std_5": 4.1,
        "disposals_last_10": 26.8,
        "disposals_std_10": 5.2,
        "disposals_last_20": 25.4,
        "disposals_std_20": 6.1,
        "disposals_prev": 30
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "over_195_prob" in data
    assert "over_245_prob" in data
    assert "over_295_prob" in data
    assert "over_195_pred" in data
    assert "over_245_pred" in data
    assert "over_295_pred" in data
    for prob in [data["over_195_prob"], data["over_245_prob"], data["over_295_prob"]]:
        assert 0.0 <= prob <= 1.0
    for pred in [data["over_195_pred"], data["over_245_pred"], data["over_295_pred"]]:
        assert pred in [0, 1]

def test_predict_missing_required_field():
    payload = {
        "year": 2026,
        "round": 5,
        "player": "Marcus Bontempelli",
        # missing opponent, venue, etc.
        "team": "Western Bulldogs"
    }
    response = client.post("/predict", json=payload)
    # Will likely fail validation or feature construction
    assert response.status_code in [400, 422]

def test_predict_invalid_type():
    payload = {
        "year": "not a number",
        "round": 5,
        "player": "Test",
        "team": "A",
        "opponent": "B",
        "venue": "C",
        "is_home": 0,
        "had_bye_last_week": 0,
        "opponent_avg_disposals": 23.5,
        "opponent_defensive_avg": 15.2,
        "venue_avg_disposals": 22.1,
        "disposals_last_5": 28.3,
        "disposals_std_5": 4.1,
        "disposals_last_10": 26.8,
        "disposals_std_10": 5.2,
        "disposals_last_20": 25.4,
        "disposals_std_20": 6.1,
        "disposals_prev": 30
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
