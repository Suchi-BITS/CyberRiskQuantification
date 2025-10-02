from bag import build_attack_graph, infer_attack_probabilities
from sm import generate_synthetic_data, train_models
from gnn import build_synthetic_graph, train_gnn
from cal import calibrate_model, explain_model
import matplotlib.pyplot as plt

import csv
import datetime
import statistics
import uuid
import os

# ---------------- Utility: CSV Loader ---------------- #
def load_csv(filename):
    """
    Load CSV into list of dicts.
    """
    if not os.path.exists(filename):
        print(f"⚠️ File not found: {filename}, returning empty list.")
        return []
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

# ---------------- Risk Results Generator ---------------- #
def calculate_risk_results(assets, vulns, controls, output_file="risk_results.csv"):
    """
    Generate risk results based on assets, vulnerabilities, and controls.
    """
    results = []

    for asset in assets:
        asset_id = asset["asset_id"]

        # Filter vulnerabilities tied to this asset
        asset_vulns = [v for v in vulns if v["asset_id"] == asset_id]

        if not asset_vulns:
            continue

        # Inherent likelihood from CVSS scores
        cvss_scores = [float(v["cvss_v3"]) for v in asset_vulns if v.get("cvss_v3")]
        inherent_likelihood = min(1.0, statistics.mean(cvss_scores) / 10.0)

        # Controls effectiveness (average score of applied controls)
        applied_controls = [c for c in controls if asset_id in c.get("asset_scope", [])]
        control_effectiveness = statistics.mean(
            [float(c["effectiveness_score"]) for c in applied_controls]
        ) if applied_controls else 0.0

        residual_likelihood = max(0.0, inherent_likelihood * (1 - control_effectiveness))

        # Impact: use business_value as proxy
        business_value = float(asset.get("business_value_usd", 100000))

        # Simple impact distribution (placeholder for Monte Carlo)
        impact_distribution = {
            "mean": business_value,
            "p10": business_value * 0.5,
            "p50": business_value,
            "p90": business_value * 1.5
        }

        expected_annual_loss_usd = residual_likelihood * business_value

        results.append({
            "risk_id": str(uuid.uuid4()),
            "asset_id": asset_id,
            "scenario_id": "default",
            "inherent_likelihood": round(inherent_likelihood, 3),
            "residual_likelihood": round(residual_likelihood, 3),
            "impact_usd_distribution_summary": impact_distribution,
            "expected_annual_loss_usd": round(expected_annual_loss_usd, 2),
            "controls_applied": [c["control_id"] for c in applied_controls],
            "timestamp": datetime.datetime.utcnow().isoformat()
        })

    # Write to CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "risk_id",
                "asset_id",
                "scenario_id",
                "inherent_likelihood",
                "residual_likelihood",
                "impact_usd_distribution_summary",
                "expected_annual_loss_usd",
                "controls_applied",
                "timestamp"
            ]
        )
        writer.writeheader()
        for row in results:
            row["impact_usd_distribution_summary"] = str(row["impact_usd_distribution_summary"])
            row["controls_applied"] = ";".join(row["controls_applied"])
            writer.writerow(row)

    print(f"✅ Generated {output_file} with {len(results)} rows.")


# ---------------- Main Pipeline ---------------- #
def main():
    print("\n--- Bayesian Attack Graph ---")
    model = build_attack_graph()
    infer_attack_probabilities(model)

    print("\n--- Supervised Models ---")
    df = generate_synthetic_data()
    lr, xgb, X_test, y_test, probs_xgb = train_models(df)

    print("\n--- Calibration & Explainability ---")
    calibrated_probs = calibrate_model(probs_xgb, y_test)
    explain_model(xgb, X_test)

    print("\n--- Graph Neural Network ---")
    data = build_synthetic_graph()
    gnn_model = train_gnn(data)

    print("\n--- Risk Results ---")
    assets = load_csv("assets.csv")
    vulns = load_csv("vulns.csv")
    controls = load_csv("controls.csv")

    calculate_risk_results(assets, vulns, controls)


if __name__ == "__main__":
    main()
