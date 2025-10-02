#!/usr/bin/env python3
"""
QuantifyXR — FAIR Monte Carlo Analysis

This script demonstrates a FAIR-style Monte Carlo calculation for Expected Annual Loss (EAL),
residual risk after controls, and includes simple ML models to predict exploit availability.
The script reads assets.csv, vulns.csv, and controls.csv from the same directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Configuration
DATA_DIR = Path('.')
N_SIMS = 10000
RANDOM_SEED = 42


def load_data():
    """Load assets, vulnerabilities, and controls data."""
    assets = pd.read_csv(DATA_DIR / 'assets.csv')
    vulns = pd.read_csv(DATA_DIR / 'vulns.csv')
    controls = pd.read_csv(DATA_DIR / 'controls.csv', converters={'asset_scope': eval})

    print(f"Loaded {assets.shape} assets; {vulns.shape} vulns; {controls.shape} controls")
    print("\nAssets preview:")
    print(assets.head())
    return assets, vulns, controls


def exposure_factor(tag):
    """Calculate exposure factor based on asset tag."""
    if tag == 'critical':
        return 2.0
    if tag == 'sensitive':
        return 1.5
    return 0.6


def combined_control_effectiveness_for_asset(asset_id, controls_df):
    """Calculate combined control effectiveness for a given asset."""
    effs = []
    for _, row in controls_df.iterrows():
        scope = row['asset_scope']
        try:
            if asset_id in scope:
                effs.append(float(row['effectiveness_score']))
        except Exception:
            continue

    if not effs:
        return 0.0

    prod = 1.0
    for e in effs:
        prod *= (1.0 - e)
    return 1.0 - prod


def breach_prob_from_cvss(cvss_v3, exploit_available, exposure_mult):
    """Calculate breach probability from CVSS score and other factors."""
    base = (float(cvss_v3) / 10.0) * 0.15
    if exploit_available:
        base *= 2.0
    base *= exposure_mult
    return min(base, 0.95)


def fair_monte_carlo(assets_df, vulns_df, controls_df, n_sims=10000, random_seed=42):
    """Run FAIR-style Monte Carlo simulation for Expected Annual Loss calculation."""
    np.random.seed(random_seed)
    results = []
    grouped = vulns_df.groupby('asset_id')

    for _, asset in assets_df.iterrows():
        aid = asset['asset_id']
        business_value = float(asset['business_value_usd'])
        tag = asset.get('tags', '')
        exp_mult = exposure_factor(tag)

        asset_vulns = grouped.get_group(aid) if aid in grouped.groups else pd.DataFrame([])
        control_eff = combined_control_effectiveness_for_asset(aid, controls_df)

        total_losses = np.zeros(n_sims)
        for _, v in asset_vulns.iterrows():
            p = breach_prob_from_cvss(v['cvss_v3'], v['exploit_available'], exp_mult)
            p_residual = p * (1.0 - control_eff)
            draws = np.random.rand(n_sims) < p_residual
            fracs = 0.01 + (np.random.beta(2, 20, size=n_sims) * (0.20 - 0.01))
            impacts = fracs * business_value
            total_losses += draws * impacts

        results.append({
            'asset_id': aid,
            'business_value_usd': business_value,
            'n_vulns': int(len(asset_vulns)),
            'control_effectiveness': control_eff,
            'eal_mean_usd': float(np.mean(total_losses)),
            'eal_p10_usd': float(np.percentile(total_losses, 10)),
            'eal_p50_usd': float(np.percentile(total_losses, 50)),
            'eal_p90_usd': float(np.percentile(total_losses, 90))
        })

    return pd.DataFrame(results)


def plot_top_assets(mc_results, top_n=10):
    """Plot top assets by Expected Annual Loss."""
    top = mc_results.sort_values('eal_mean_usd', ascending=False).head(top_n)

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(top)), top['eal_mean_usd'])
    plt.xticks(range(len(top)), top['asset_id'], rotation=45)
    plt.ylabel('EAL (USD)')
    plt.title(f'Top {top_n} assets by Expected Annual Loss (mean)')
    plt.tight_layout()
    plt.savefig('top_assets_eal.png', dpi=150, bbox_inches='tight')
    print("Plot saved to top_assets_eal.png")
    plt.show()


def train_ml_models(assets, vulns):
    """Train ML models to predict exploit availability."""
    v = vulns.copy()
    a = assets[['asset_id', 'business_value_usd', 'tags']]
    df = v.merge(a, left_on='asset_id', right_on='asset_id', how='left')

    df['cvss_v3'] = df['cvss_v3'].astype(float)
    df['tag_score'] = df['tags'].map({'critical': 2, 'sensitive': 1, 'low-priority': 0}).fillna(0)

    X = df[['cvss_v3', 'business_value_usd', 'tag_score']]
    X['log_business_value'] = np.log1p(X['business_value_usd'])
    X = X[['cvss_v3', 'log_business_value', 'tag_score']]
    y = df['exploit_available'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    probs_lr = lr.predict_proba(X_test)[:, 1]
    preds_lr = lr.predict(X_test)

    print("\n" + "=" * 60)
    print("Logistic Regression Classification Report")
    print("=" * 60)
    print(classification_report(y_test, preds_lr))
    print(f"LR ROC AUC: {roc_auc_score(y_test, probs_lr):.4f}")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    probs_rf = rf.predict_proba(X_test)[:, 1]
    preds_rf = rf.predict(X_test)

    print("\n" + "=" * 60)
    print("Random Forest Classification Report")
    print("=" * 60)
    print(classification_report(y_test, preds_rf))
    print(f"RF ROC AUC: {roc_auc_score(y_test, probs_rf):.4f}")

    return lr, rf


def main():
    """Main execution function."""
    print("=" * 60)
    print("QuantifyXR — FAIR Monte Carlo Analysis")
    print("=" * 60)

    # Load data
    assets, vulns, controls = load_data()

    # Run Monte Carlo simulation
    print("\n" + "=" * 60)
    print("Running FAIR Monte Carlo Simulation...")
    print("=" * 60)
    mc_results = fair_monte_carlo(assets, vulns, controls, n_sims=N_SIMS, random_seed=RANDOM_SEED)

    print("\nTop 10 Assets by Expected Annual Loss")
    print(mc_results.sort_values('eal_mean_usd', ascending=False).head(10))

    # Save results
    mc_results.to_csv('monte_carlo_results.csv', index=False)
    print("\nResults saved to monte_carlo_results.csv")

    # Plot results
    plot_top_assets(mc_results)

    # Train ML models
    print("\n" + "=" * 60)
    print("Training ML Models for Exploit Prediction...")
    print("=" * 60)
    lr_model, rf_model = train_ml_models(assets, vulns)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
