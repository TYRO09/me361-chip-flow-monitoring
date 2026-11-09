"""Task 6: literature panels: Colwell / Young-Mathew-Oxley direction and the observed-chip-form table."""
import os
import sys

import pytest
from starlette.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, ROOT)

from app import server  # noqa: E402

# printed by code/physics_flow_angle.py and used in the report (r_eps 0.4, kappa_r 95, eps_r 80)
REPORT = {(0.08, 0.9): (23.46, 11.64), (0.15, 0.9): (25.86, 12.76), (0.08, 0.3): (55.09, 44.88),
          (0.15, 0.3): (57.94, 48.66), (0.08, 0.6): (34.93, 20.97), (0.15, 0.6): (37.77, 22.85)}


@pytest.fixture()
def client():
    with TestClient(server.app) as c:
        yield c


@pytest.mark.parametrize("fa,vals", list(REPORT.items()))
def test_theory_matches_report_values(client, fa, vals):
    f, ap = fa
    j = client.get(f"/api/theory?f={f}&ap={ap}&r_eps=0.4").json()
    assert j["eta_colwell"] == pytest.approx(vals[0], abs=0.1) and j["eta_ymo"] == pytest.approx(vals[1], abs=0.1)
    assert j["psi_theory_colwell"] == pytest.approx(180 - vals[0], abs=0.1)     # frame conversion used in the report
    assert j["psi_theory_ymo"] == pytest.approx(180 - vals[1], abs=0.1)


def test_theory_limit_deep_cut_goes_to_minus_five(client):
    j = client.get("/api/theory?f=0.1&ap=20&r_eps=0.4").json()
    assert j["eta_colwell"] == pytest.approx(-5.0, abs=2.0)           # 90 - kappa_r for a 95-degree holder


def test_theory_rejects_nonsense(client):
    assert client.get("/api/theory?f=0&ap=0.5").status_code == 422
    assert client.get("/api/theory?f=abc&ap=0.5").status_code == 422


def test_chip_forms_table_covers_all_runs(client):
    rows = client.get("/api/chip_forms").json()["runs"]
    assert [r["run"] for r in rows] == ["R1", "R2", "R3", "R4", "R5", "R6"]
    r2 = next(r for r in rows if r["run"] == "R2")
    assert r2["tangled"] is True and sum(r["tangled"] for r in rows) == 1
    assert all(r["f"] > 0 and r["ap"] > 0 and r["observed"] for r in rows)
