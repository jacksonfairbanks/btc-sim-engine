"""
Export results to CSV, JSON, and HTML.
"""
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from validation.scorer import ScoreCard


def _today() -> str:
    """Return today's date as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


class ResultsExporter:
    """Export leaderboard results to various formats."""

    def __init__(self, results: list[ScoreCard], output_dir: str = "results"):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _to_records(self) -> list[dict[str, Any]]:
        """Flatten ScoreCards to dicts."""
        records = []
        for card in self.results:
            row = {
                "rank": card.rank,
                "model": card.model_name,
                "horizon": card.horizon,
                "composite_score": round(card.composite_score, 6),
            }
            for k, v in card.scoring_metrics.items():
                row[f"score_{k}"] = round(v, 6)
            for k, v in card.params.items():
                row[f"param_{k}"] = v
            records.append(row)
        return records

    def to_csv(self, filename: str | None = None) -> Path:
        """Export to CSV."""
        records = self._to_records()
        path = self.output_dir / (filename or f"leaderboard_{_today()}.csv")

        if not records:
            return path

        # Collect all fieldnames across all records (models have different params)
        all_fields = []
        seen = set()
        for rec in records:
            for k in rec.keys():
                if k not in seen:
                    all_fields.append(k)
                    seen.add(k)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

        return path

    def to_json(self, filename: str | None = None) -> Path:
        """Export to JSON."""
        records = self._to_records()
        path = self.output_dir / (filename or f"leaderboard_{_today()}.json")

        with open(path, "w") as f:
            json.dump(records, f, indent=2, default=str)

        return path

    def to_html(self, filename: str | None = None) -> Path:
        """Export to standalone HTML with dark theme."""
        records = self._to_records()
        path = self.output_dir / (filename or f"leaderboard_{_today()}.html")

        if not records:
            path.write_text("<html><body>No results</body></html>")
            return path

        # Collect all headers across all records
        headers = []
        seen = set()
        for rec in records:
            for k in rec.keys():
                if k not in seen:
                    headers.append(k)
                    seen.add(k)

        rows_html = ""
        for row in records:
            cells = ""
            for h in headers:
                val = row.get(h, "")
                if isinstance(val, float):
                    val = f"{val:.4f}"
                cells += f"<td>{val}</td>"
            rows_html += f"<tr>{cells}</tr>\n"

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>BTC Sim Engine — Leaderboard ({_today()})</title>
<style>
  body {{
    background: #0a0a0a;
    color: #e0e0e0;
    font-family: 'JetBrains Mono', 'IBM Plex Mono', monospace;
    padding: 2rem;
  }}
  h1 {{
    color: #f7931a;
    font-size: 1.4rem;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin-top: 1rem;
  }}
  th {{
    background: #1a1a1a;
    color: #f7931a;
    padding: 8px 12px;
    text-align: left;
    font-size: 0.8rem;
    border-bottom: 2px solid #333;
  }}
  td {{
    padding: 6px 12px;
    border-bottom: 1px solid #1a1a1a;
    font-size: 0.8rem;
  }}
  tr:hover {{
    background: #111111;
  }}
  .footer {{
    margin-top: 2rem;
    color: #666;
    font-size: 0.7rem;
  }}
</style>
</head>
<body>
<h1>BTC Price Path Simulation — Leaderboard ({_today()})</h1>
<table>
<thead><tr>{"".join(f"<th>{h}</th>" for h in headers)}</tr></thead>
<tbody>
{rows_html}
</tbody>
</table>
<div class="footer">@LongGamma — btc-sim-engine</div>
</body>
</html>"""

        path.write_text(html)
        return path

    def export_all(self) -> dict[str, Path]:
        """Export to all configured formats."""
        return {
            "csv": self.to_csv(),
            "json": self.to_json(),
            "html": self.to_html(),
        }
