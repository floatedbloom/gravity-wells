import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from scipy.stats import spearmanr


def load_bias_rows(bias_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(bias_dir.glob("bias_values_*.json")):
        subreddit = path.stem.replace("bias_values_", "")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            author = item.get("author")
            bias = item.get("bias_seeking")
            label = item.get("dogmatism_label")
            if author is None or bias is None or label is None:
                continue
            rows.append({
                "subreddit": subreddit,
                "author": author,
                "bias_seeking": bias,
                "dogmatism_label": label,
            })
    return rows


def map_label_to_score(label: str) -> float:
    mapping: Dict[str, float] = {
        "Deeply Rooted": 1.0,
        "Firm but Open": 2.0 / 3.0,
        "Open to Dialogue": 1.0 / 3.0,
        "Flexible": 0.0,
    }
    # Default for unexpected labels: midpoint
    return float(mapping.get(label, 0.5))


def compute_spearman(df: pd.DataFrame) -> Tuple[float, float, int]:
    if df.shape[0] < 5:
        return float("nan"), float("nan"), int(df.shape[0])
    rho, p = spearmanr(df["bias_seeking"].to_numpy(), df["dogmatism_score"].to_numpy())
    return float(rho), float(p), int(df.shape[0])


def main() -> None:
    root = Path(__file__).resolve().parent
    bias_dir = root / "data" / "bias"

    rows = load_bias_rows(bias_dir)
    if not rows:
        print("No rows loaded from bias files.")
        return

    df = pd.DataFrame(rows)
    df["dogmatism_score"] = df["dogmatism_label"].map(map_label_to_score)

    results: List[Dict] = []
    # Per-subreddit
    for subreddit, sub_df in sorted(df.groupby("subreddit")):
        rho, p, n = compute_spearman(sub_df)
        results.append({
            "scope": "per_subreddit",
            "subreddit": subreddit,
            "n": n,
            "spearman_rho": rho,
            "p_value": p,
        })

    # Overall
    rho_all, p_all, n_all = compute_spearman(df)
    results.append({
        "scope": "overall",
        "subreddit": "ALL",
        "n": n_all,
        "spearman_rho": rho_all,
        "p_value": p_all,
    })

    out_path = root / "bias_vs_dogmatism_spearman.csv"
    pd.DataFrame(results).sort_values(["scope", "spearman_rho"], ascending=[True, False]).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()



