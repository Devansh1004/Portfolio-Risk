import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

def make_chart(fig_func, path, figsize=(7, 4)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    fig_func(ax)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

def dict_table(data, title, styles, col_widths=None):
    rows = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in data.items()]
    table = Table([[title, ""]] + rows, colWidths=col_widths or [250, 150])
    table.setStyle(TableStyle([
        ("SPAN", (0, 0), (1, 0)),
        ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#1f3864")),
        ("TEXTCOLOR", (0, 0), (1, 0), colors.white),
        ("FONTNAME", (0, 0), (1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9)
    ]))
    return table

def generate_report(output_path, portfolio_returns, asset_returns, weights, risk_metrics, var_results, stress_results, decomposition_df, backtest, exceptions, correlation_matrix):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Portfolio Risk Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Assets: {', '.join(asset_returns.columns)}", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Risk Metrics", styles["Heading2"]))
    story.append(dict_table(risk_metrics, "Metric / Value", styles))
    story.append(Spacer(1, 12))
    story.append(Paragraph("VaR Comparison (95%, 1-day)", styles["Heading2"]))
    var_flat = {}
    for method, vals in var_results.items():
        for k, v in vals.items():
            var_flat[f"{method}_{k}"] = v
    story.append(dict_table(var_flat, "Method_Metric / Value", styles))
    story.append(PageBreak())
    make_chart(lambda ax: ax.plot((1 + portfolio_returns).cumprod()), "plots/report_cum_returns.png")
    story.append(Paragraph("Cumulative Portfolio Returns", styles["Heading2"]))
    story.append(Image("plots/report_cum_returns.png", width=6.5 * inch, height=3.5 * inch))
    story.append(Spacer(1, 12))
    make_chart(lambda ax: ax.plot(portfolio_returns.rolling(21).std()), "plots/report_rolling_vol.png")
    story.append(Paragraph("Rolling 21-Day Volatility", styles["Heading2"]))
    story.append(Image("plots/report_rolling_vol.png", width=6.5 * inch, height=3.5 * inch))
    story.append(PageBreak())
    def corr_heatmap(ax):
        im = ax.imshow(correlation_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45)
        ax.set_yticklabels(correlation_matrix.columns)
        plt.colorbar(im, ax=ax)
    make_chart(corr_heatmap, "plots/report_correlation.png")
    story.append(Paragraph("Asset Correlation Matrix", styles["Heading2"]))
    story.append(Image("plots/report_correlation.png", width=5.5 * inch, height=4.5 * inch))
    story.append(PageBreak())
    def decomp_chart(ax):
        ax.barh(decomposition_df.index, decomposition_df["percent_of_risk"])
        ax.set_xlabel("Percent of Portfolio Risk")
    make_chart(decomp_chart, "plots/report_decomposition.png")
    story.append(Paragraph("Risk Decomposition by Asset", styles["Heading2"]))
    story.append(Image("plots/report_decomposition.png", width=6.5 * inch, height=3.5 * inch))
    story.append(Spacer(1, 12))
    story.append(Table([["Asset"] + list(decomposition_df.columns)] + [[i] + [f"{v:.4f}" for v in row] for i, row in decomposition_df.iterrows()], style=TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)])))
    story.append(PageBreak())
    story.append(Paragraph("Stress Test Scenarios", styles["Heading2"]))
    for scenario in stress_results:
        story.append(dict_table(scenario, "Scenario / Value", styles))
        story.append(Spacer(1, 8))
    story.append(PageBreak())
    story.append(Paragraph("VaR Backtesting (Kupiec Test)", styles["Heading2"]))
    story.append(dict_table(backtest, "Metric / Value", styles))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Total VaR breaches: {exceptions['total_breaches']}, consecutive breach pairs: {exceptions['consecutive_breach_pairs']}", styles["Normal"]))
    doc.build(story)
    return output_path
