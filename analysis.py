"""
AgentDeception Results Analysis

Run this script after evals complete to generate interactive charts.

Usage:
    python analysis.py [log_file]

If no log file specified, uses the most recent one.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Try to import plotting libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Install plotly for interactive charts: pip install plotly")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Install pandas for data analysis: pip install pandas")


def load_eval_log(log_path: Path):
    """Load and parse an Inspect AI eval log file."""
    from inspect_ai.log import read_eval_log
    return read_eval_log(str(log_path))


def extract_sample_data(log) -> list[dict]:
    """Extract per-sample results from eval log."""
    samples = []

    for sample in log.samples or []:
        metadata = sample.metadata or {}
        scores = sample.scores or {}

        # Get deception scores
        compliance = scores.get("compliance_scorer")
        deception = scores.get("deception_scorer")

        compliance_val = compliance.value if compliance else 0
        deception_val = deception.value if deception else 0
        deception_answer = deception.answer if deception else "unknown"
        deception_meta = deception.metadata if deception else {}

        samples.append({
            "id": sample.id or "",
            "category": metadata.get("category", "unknown"),
            "severity": metadata.get("severity", 0),
            "instruction_level": metadata.get("instruction_level", "unknown"),
            "has_privacy_instruction": metadata.get("has_privacy_instruction", False),
            "complied": compliance_val == 1.0,
            "deception_type": deception_answer,
            "deception_score": deception_val,
            "acted_covertly": deception_meta.get("acted_without_permission", False) if deception_meta else False,
            "tools_used": deception_meta.get("tools_used", []) if deception_meta else [],
        })

    return samples


def print_summary(samples: list[dict]):
    """Print text summary of results."""
    total = len(samples)
    complied = sum(1 for s in samples if s["complied"])
    acted_covertly = sum(1 for s in samples if s["acted_covertly"])

    # Deception type counts
    deception_counts = defaultdict(int)
    for s in samples:
        deception_counts[s["deception_type"]] += 1

    print(f"\n{'='*60}")
    print(f"AGENTDECEPTION RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Complied (didn't use tools): {complied} ({100*complied/total:.1f}%)")
    print(f"Acted covertly: {acted_covertly} ({100*acted_covertly/total:.1f}%)")

    print(f"\n--- Deception Categories ---")
    for dtype, count in sorted(deception_counts.items(), key=lambda x: -x[1]):
        print(f"{dtype}: {count} ({100*count/total:.1f}%)")

    # By severity
    print(f"\n--- Covert Action Rate by Severity ---")
    for sev in range(1, 6):
        sev_samples = [s for s in samples if s["severity"] == sev]
        if sev_samples:
            sev_covert = sum(1 for s in sev_samples if s["acted_covertly"])
            print(f"Severity {sev}: {sev_covert}/{len(sev_samples)} ({100*sev_covert/len(sev_samples):.1f}%)")

    # By privacy instruction
    print(f"\n--- By Privacy Instruction ---")
    with_privacy = [s for s in samples if s["has_privacy_instruction"]]
    without_privacy = [s for s in samples if not s["has_privacy_instruction"]]

    if with_privacy:
        wp_covert = sum(1 for s in with_privacy if s["acted_covertly"])
        print(f"With privacy instruction: {wp_covert}/{len(with_privacy)} acted covertly ({100*wp_covert/len(with_privacy):.1f}%)")

    if without_privacy:
        wop_covert = sum(1 for s in without_privacy if s["acted_covertly"])
        print(f"Without privacy instruction: {wop_covert}/{len(without_privacy)} acted covertly ({100*wop_covert/len(without_privacy):.1f}%)")

    # Lying analysis
    print(f"\n--- Lying Analysis (among those who acted covertly) ---")
    covert_samples = [s for s in samples if s["acted_covertly"]]
    if covert_samples:
        lied = sum(1 for s in covert_samples if s["deception_type"] == "lied_acted")
        admitted = sum(1 for s in covert_samples if s["deception_type"] == "honest_defiant")
        deflected = sum(1 for s in covert_samples if s["deception_type"] == "deflected")
        print(f"Total who acted covertly: {len(covert_samples)}")
        print(f"  - Lied about it: {lied} ({100*lied/len(covert_samples):.1f}%)")
        print(f"  - Admitted it: {admitted} ({100*admitted/len(covert_samples):.1f}%)")
        print(f"  - Deflected: {deflected} ({100*deflected/len(covert_samples):.1f}%)")
    else:
        print("No samples with covert action")


def create_charts(samples: list[dict], output_dir: Path):
    """Create interactive charts with Plotly."""
    if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
        print("Skipping charts (install plotly and pandas)")
        return

    df = pd.DataFrame(samples)
    output_dir.mkdir(exist_ok=True)

    # 1. Deception Type Distribution (pie chart)
    deception_counts = df["deception_type"].value_counts().reset_index()
    deception_counts.columns = ["type", "count"]

    color_map = {
        "honest_compliant": "#2ecc71",  # Green
        "honest_defiant": "#3498db",    # Blue
        "deflected": "#f39c12",         # Orange
        "lied_acted": "#e74c3c",        # Red
        "lied_didnt_act": "#9b59b6",    # Purple
        "unknown": "#95a5a6"            # Gray
    }

    fig1 = px.pie(
        deception_counts,
        values="count",
        names="type",
        title="Deception Type Distribution",
        color="type",
        color_discrete_map=color_map
    )
    fig1.write_html(output_dir / "deception_types.html")
    print(f"Saved: {output_dir / 'deception_types.html'}")

    # 2. Covert Action Rate by Severity
    severity_data = df.groupby("severity").agg({
        "acted_covertly": ["sum", "count"]
    }).reset_index()
    severity_data.columns = ["severity", "covert", "total"]
    severity_data["rate"] = severity_data["covert"] / severity_data["total"] * 100

    fig2 = px.bar(
        severity_data,
        x="severity",
        y="rate",
        title="Covert Action Rate by Severity Level",
        labels={"severity": "Severity (1=mild, 5=critical)", "rate": "Covert Action Rate (%)"},
        color="rate",
        color_continuous_scale="Reds"
    )
    fig2.update_layout(showlegend=False)
    fig2.write_html(output_dir / "covert_by_severity.html")
    print(f"Saved: {output_dir / 'covert_by_severity.html'}")

    # 3. Privacy Instruction Effect
    privacy_data = df.groupby("has_privacy_instruction").agg({
        "acted_covertly": ["sum", "count"],
        "deception_score": "mean"
    }).reset_index()
    privacy_data.columns = ["has_privacy", "covert", "total", "avg_score"]
    privacy_data["covert_rate"] = privacy_data["covert"] / privacy_data["total"] * 100
    privacy_data["label"] = privacy_data["has_privacy"].map({True: "With Privacy Instruction", False: "Without Privacy Instruction"})

    fig3 = px.bar(
        privacy_data,
        x="label",
        y="covert_rate",
        title="Effect of Privacy Instruction on Covert Action",
        labels={"label": "", "covert_rate": "Covert Action Rate (%)"},
        color="label",
        color_discrete_sequence=["#3498db", "#e74c3c"]
    )
    fig3.update_layout(showlegend=False)
    fig3.write_html(output_dir / "privacy_instruction_effect.html")
    print(f"Saved: {output_dir / 'privacy_instruction_effect.html'}")

    # 4. Deception by Category
    category_deception = df.groupby(["category", "deception_type"]).size().reset_index(name="count")

    fig4 = px.bar(
        category_deception,
        x="category",
        y="count",
        color="deception_type",
        title="Deception Types by Category",
        labels={"category": "Category", "count": "Count", "deception_type": "Deception Type"},
        color_discrete_map=color_map,
        barmode="stack"
    )
    fig4.update_xaxes(tickangle=45)
    fig4.write_html(output_dir / "deception_by_category.html")
    print(f"Saved: {output_dir / 'deception_by_category.html'}")

    # 5. Heatmap: Category x Severity (Covert Action Rate)
    heatmap_data = df.groupby(["category", "severity"]).agg({
        "acted_covertly": "mean"
    }).reset_index()
    heatmap_data["acted_covertly"] = heatmap_data["acted_covertly"] * 100

    heatmap_pivot = heatmap_data.pivot(index="category", columns="severity", values="acted_covertly")

    fig5 = px.imshow(
        heatmap_pivot,
        title="Covert Action Rate Heatmap: Category × Severity",
        labels={"x": "Severity", "y": "Category", "color": "Covert Action Rate (%)"},
        color_continuous_scale="RdYlGn_r",
        aspect="auto"
    )
    fig5.write_html(output_dir / "covert_heatmap.html")
    print(f"Saved: {output_dir / 'covert_heatmap.html'}")

    # 6. Lying Rate Among Covert Actors by Severity
    covert_df = df[df["acted_covertly"] == True].copy()
    if len(covert_df) > 0:
        covert_df["lied"] = covert_df["deception_type"] == "lied_acted"
        lying_by_severity = covert_df.groupby("severity").agg({
            "lied": ["sum", "count"]
        }).reset_index()
        lying_by_severity.columns = ["severity", "lied", "total"]
        lying_by_severity["rate"] = lying_by_severity["lied"] / lying_by_severity["total"] * 100

        fig6 = px.bar(
            lying_by_severity,
            x="severity",
            y="rate",
            title="Lying Rate by Severity (Among Those Who Acted Covertly)",
            labels={"severity": "Severity", "rate": "Lying Rate (%)"},
            color="rate",
            color_continuous_scale="Reds"
        )
        fig6.write_html(output_dir / "lying_by_severity.html")
        print(f"Saved: {output_dir / 'lying_by_severity.html'}")

    # 7. Combined Dashboard
    fig7 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Deception Types",
            "Covert Action by Severity",
            "Privacy Instruction Effect",
            "Category × Severity Heatmap"
        ),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "heatmap"}]]
    )

    # Add pie chart
    fig7.add_trace(
        go.Pie(labels=deception_counts["type"], values=deception_counts["count"],
               marker_colors=[color_map.get(t, "#95a5a6") for t in deception_counts["type"]]),
        row=1, col=1
    )

    # Add severity bar
    fig7.add_trace(
        go.Bar(x=severity_data["severity"], y=severity_data["rate"], name="Covert Rate",
               marker_color="indianred"),
        row=1, col=2
    )

    # Add privacy comparison
    fig7.add_trace(
        go.Bar(x=privacy_data["label"], y=privacy_data["covert_rate"], name="By Privacy",
               marker_color=["#3498db", "#e74c3c"]),
        row=2, col=1
    )

    # Add heatmap
    fig7.add_trace(
        go.Heatmap(z=heatmap_pivot.values, x=heatmap_pivot.columns.tolist(),
                   y=heatmap_pivot.index.tolist(), colorscale="RdYlGn_r"),
        row=2, col=2
    )

    fig7.update_layout(height=900, title_text="AgentDeception Results Dashboard", showlegend=False)
    fig7.write_html(output_dir / "dashboard.html")
    print(f"Saved: {output_dir / 'dashboard.html'}")


def main():
    # Find log file
    logs_dir = Path(__file__).parent / "logs"

    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        # Use most recent log
        log_files = sorted(logs_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            print("No eval logs found in logs/")
            return
        log_path = log_files[0]

    print(f"Loading: {log_path}")
    log = load_eval_log(log_path)
    samples = extract_sample_data(log)

    if not samples:
        print("No samples found in log")
        return

    print_summary(samples)

    # Create charts
    charts_dir = Path(__file__).parent / "charts"
    create_charts(samples, charts_dir)

    print(f"\nOpen charts in browser: file://{charts_dir.absolute()}/dashboard.html")


if __name__ == "__main__":
    main()
