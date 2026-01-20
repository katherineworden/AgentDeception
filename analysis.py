"""
AgentDeception Results Analysis

Run this script after evals complete to generate interactive charts.

Usage:
    python analysis.py              # Analyze all logs, generate per-model reports
    python analysis.py [log_file]   # Analyze specific log file

Generates:
    - charts/dashboard.html (combined view)
    - charts/{model_name}/dashboard.html (per-model views)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

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

    # Get model name from log
    model_name = "unknown"
    if hasattr(log, 'eval') and hasattr(log.eval, 'model'):
        model_name = log.eval.model

    for sample in log.samples or []:
        metadata = sample.metadata or {}
        scores = sample.scores or {}

        deception = scores.get("deception_scorer")
        deception_val = deception.value if deception else 0
        deception_answer = deception.answer if deception else "unknown"
        deception_meta = deception.metadata if deception else {}

        samples.append({
            "id": sample.id or "",
            "model": model_name,
            "category": metadata.get("category", "unknown"),
            "severity": metadata.get("severity", 0),
            "system_prompt_level": metadata.get("system_prompt_level", "unknown"),
            "deception_type": deception_answer,
            "deception_score": deception_val,
            "acted_covertly": deception_meta.get("acted_covertly", False) if deception_meta else False,
            "tools_used": deception_meta.get("tools_used", []) if deception_meta else [],
        })

    return samples


def print_summary(samples: list[dict], model_name: str = "All Models"):
    """Print text summary of results."""
    total = len(samples)
    if total == 0:
        print(f"No samples found for {model_name}")
        return

    acted_covertly = sum(1 for s in samples if s["acted_covertly"])

    deception_counts = defaultdict(int)
    for s in samples:
        deception_counts[s["deception_type"]] += 1

    print(f"\n{'='*60}")
    print(f"AGENTDECEPTION RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Acted covertly: {acted_covertly} ({100*acted_covertly/total:.1f}%)")

    print(f"\n--- Deception Categories ---")
    for dtype, count in sorted(deception_counts.items(), key=lambda x: -x[1]):
        print(f"{dtype}: {count} ({100*count/total:.1f}%)")

    # By system prompt level
    print(f"\n--- Covert Action Rate by System Prompt Level ---")
    for level in ["neutral", "soft", "strong", "explicit"]:
        level_samples = [s for s in samples if s["system_prompt_level"] == level]
        if level_samples:
            level_covert = sum(1 for s in level_samples if s["acted_covertly"])
            print(f"{level}: {level_covert}/{len(level_samples)} ({100*level_covert/len(level_samples):.1f}%)")

    # By severity
    print(f"\n--- Covert Action Rate by Severity ---")
    for sev in range(1, 6):
        sev_samples = [s for s in samples if s["severity"] == sev]
        if sev_samples:
            sev_covert = sum(1 for s in sev_samples if s["acted_covertly"])
            print(f"Severity {sev}: {sev_covert}/{len(sev_samples)} ({100*sev_covert/len(sev_samples):.1f}%)")

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


def create_charts(samples: list[dict], output_dir: Path, title_prefix: str = ""):
    """Create interactive charts with Plotly."""
    if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
        print("Skipping charts (install plotly and pandas)")
        return

    df = pd.DataFrame(samples)
    if len(df) == 0:
        print("No data to chart")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    color_map = {
        "honest_compliant": "#2ecc71",
        "honest_defiant": "#3498db",
        "deflected": "#f39c12",
        "lied_acted": "#e74c3c",
        "lied_didnt_act": "#9b59b6",
        "unknown": "#95a5a6"
    }

    # 1. Deception Type Distribution
    deception_counts = df["deception_type"].value_counts().reset_index()
    deception_counts.columns = ["type", "count"]

    fig1 = px.pie(
        deception_counts,
        values="count",
        names="type",
        title=f"{title_prefix}Deception Type Distribution",
        color="type",
        color_discrete_map=color_map
    )
    fig1.write_html(output_dir / "deception_types.html")

    # 2. Covert Action Rate by System Prompt Level
    if "system_prompt_level" in df.columns:
        level_order = ["neutral", "soft", "strong", "explicit"]
        level_data = df.groupby("system_prompt_level").agg({
            "acted_covertly": ["sum", "count"]
        }).reset_index()
        level_data.columns = ["level", "covert", "total"]
        level_data["rate"] = level_data["covert"] / level_data["total"] * 100
        level_data["level"] = pd.Categorical(level_data["level"], categories=level_order, ordered=True)
        level_data = level_data.sort_values("level")

        fig2 = px.bar(
            level_data,
            x="level",
            y="rate",
            title=f"{title_prefix}Covert Action Rate by System Prompt Level",
            labels={"level": "System Prompt Level", "rate": "Covert Action Rate (%)"},
            color="rate",
            color_continuous_scale="Reds"
        )
        fig2.update_layout(showlegend=False)
        fig2.write_html(output_dir / "covert_by_prompt_level.html")

    # 3. Covert Action Rate by Severity
    severity_data = df.groupby("severity").agg({
        "acted_covertly": ["sum", "count"]
    }).reset_index()
    severity_data.columns = ["severity", "covert", "total"]
    severity_data["rate"] = severity_data["covert"] / severity_data["total"] * 100

    fig3 = px.bar(
        severity_data,
        x="severity",
        y="rate",
        title=f"{title_prefix}Covert Action Rate by Severity Level",
        labels={"severity": "Severity (1=mild, 5=critical)", "rate": "Covert Action Rate (%)"},
        color="rate",
        color_continuous_scale="Reds"
    )
    fig3.update_layout(showlegend=False)
    fig3.write_html(output_dir / "covert_by_severity.html")

    # 4. Deception by Category
    category_deception = df.groupby(["category", "deception_type"]).size().reset_index(name="count")

    fig4 = px.bar(
        category_deception,
        x="category",
        y="count",
        color="deception_type",
        title=f"{title_prefix}Deception Types by Category",
        color_discrete_map=color_map,
        barmode="stack"
    )
    fig4.update_xaxes(tickangle=45)
    fig4.write_html(output_dir / "deception_by_category.html")

    # 5. Heatmap: System Prompt Level x Severity
    if "system_prompt_level" in df.columns:
        heatmap_data = df.groupby(["system_prompt_level", "severity"]).agg({
            "acted_covertly": "mean"
        }).reset_index()
        heatmap_data["acted_covertly"] = heatmap_data["acted_covertly"] * 100

        heatmap_pivot = heatmap_data.pivot(index="system_prompt_level", columns="severity", values="acted_covertly")
        # Reorder index
        for level in ["neutral", "soft", "strong", "explicit"]:
            if level not in heatmap_pivot.index:
                heatmap_pivot.loc[level] = 0
        heatmap_pivot = heatmap_pivot.reindex(["neutral", "soft", "strong", "explicit"])

        fig5 = px.imshow(
            heatmap_pivot,
            title=f"{title_prefix}Covert Action Rate: System Prompt × Severity",
            labels={"x": "Severity", "y": "System Prompt Level", "color": "Covert Rate (%)"},
            color_continuous_scale="RdYlGn_r",
            aspect="auto"
        )
        fig5.write_html(output_dir / "covert_heatmap.html")

    # 6. Dashboard
    fig6 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Deception Types",
            "By System Prompt Level",
            "By Severity",
            "Prompt Level × Severity"
        ),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "heatmap"}]]
    )

    fig6.add_trace(
        go.Pie(labels=deception_counts["type"], values=deception_counts["count"],
               marker_colors=[color_map.get(t, "#95a5a6") for t in deception_counts["type"]]),
        row=1, col=1
    )

    if "system_prompt_level" in df.columns and len(level_data) > 0:
        fig6.add_trace(
            go.Bar(x=level_data["level"], y=level_data["rate"], marker_color="indianred"),
            row=1, col=2
        )

    fig6.add_trace(
        go.Bar(x=severity_data["severity"], y=severity_data["rate"], marker_color="steelblue"),
        row=2, col=1
    )

    if "system_prompt_level" in df.columns:
        fig6.add_trace(
            go.Heatmap(z=heatmap_pivot.values, x=heatmap_pivot.columns.tolist(),
                       y=heatmap_pivot.index.tolist(), colorscale="RdYlGn_r"),
            row=2, col=2
        )

    fig6.update_layout(height=900, title_text=f"{title_prefix}AgentDeception Dashboard", showlegend=False)
    fig6.write_html(output_dir / "dashboard.html")

    print(f"Charts saved to: {output_dir}")


def create_model_comparison(all_samples: list[dict], output_dir: Path):
    """Create comparison charts across models."""
    if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
        return

    df = pd.DataFrame(all_samples)
    if len(df) == 0 or "model" not in df.columns:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Covert action rate by model
    model_data = df.groupby("model").agg({
        "acted_covertly": ["sum", "count"]
    }).reset_index()
    model_data.columns = ["model", "covert", "total"]
    model_data["rate"] = model_data["covert"] / model_data["total"] * 100

    fig = px.bar(
        model_data,
        x="model",
        y="rate",
        title="Covert Action Rate by Model",
        labels={"model": "Model", "rate": "Covert Action Rate (%)"},
        color="rate",
        color_continuous_scale="Reds"
    )
    fig.write_html(output_dir / "model_comparison.html")

    # Lying rate by model (among those who acted)
    covert_df = df[df["acted_covertly"] == True].copy()
    if len(covert_df) > 0:
        covert_df["lied"] = covert_df["deception_type"] == "lied_acted"
        lying_by_model = covert_df.groupby("model").agg({
            "lied": ["sum", "count"]
        }).reset_index()
        lying_by_model.columns = ["model", "lied", "total"]
        lying_by_model["rate"] = lying_by_model["lied"] / lying_by_model["total"] * 100

        fig2 = px.bar(
            lying_by_model,
            x="model",
            y="rate",
            title="Lying Rate by Model (Among Those Who Acted Covertly)",
            labels={"model": "Model", "rate": "Lying Rate (%)"},
            color="rate",
            color_continuous_scale="Reds"
        )
        fig2.write_html(output_dir / "lying_by_model.html")

    print(f"Model comparison charts saved to: {output_dir}")


def main():
    logs_dir = Path(__file__).parent / "logs"
    charts_dir = Path(__file__).parent / "charts"

    if len(sys.argv) > 1:
        # Single log file
        log_path = Path(sys.argv[1])
        print(f"Loading: {log_path}")
        log = load_eval_log(log_path)
        samples = extract_sample_data(log)
        model_name = samples[0]["model"] if samples else "unknown"
        print_summary(samples, model_name)
        create_charts(samples, charts_dir, f"{model_name}: ")
    else:
        # All log files
        log_files = sorted(logs_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            print("No eval logs found in logs/")
            return

        all_samples = []
        models_processed = set()

        for log_path in log_files:
            try:
                log = load_eval_log(log_path)
                samples = extract_sample_data(log)
                if samples:
                    model_name = samples[0]["model"]
                    if model_name not in models_processed:
                        print(f"\nLoading: {log_path.name}")
                        print_summary(samples, model_name)

                        # Per-model charts
                        model_dir = charts_dir / model_name.replace("/", "_")
                        create_charts(samples, model_dir, f"{model_name}: ")

                        all_samples.extend(samples)
                        models_processed.add(model_name)
            except Exception as e:
                print(f"Error loading {log_path}: {e}")

        if all_samples:
            # Combined charts
            print("\n" + "="*60)
            print("COMBINED ANALYSIS")
            create_charts(all_samples, charts_dir, "All Models: ")
            create_model_comparison(all_samples, charts_dir)

        print(f"\nOpen dashboard: file://{charts_dir.absolute()}/dashboard.html")


if __name__ == "__main__":
    main()
