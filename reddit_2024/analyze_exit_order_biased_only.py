import json
import gzip
from pathlib import Path
from operator import itemgetter
from typing import Dict, List, Set

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


def load_bias_authors(bias_path: Path) -> Set[str]:
    authors: Set[str] = set()
    try:
        data = json.loads(bias_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "author" in item:
                    authors.add(item["author"])
    except Exception:
        pass
    return authors


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # Inputs created by other scripts
    auths_path = repo_root / 'auths_by_subr-sorted.json.gz'
    agents_path = repo_root / 'agents_by_subr-par_sorted.json.gz'

    auths_by_subr: Dict[str, Dict[str, List[int]]] = json.load(gzip.open(auths_path, 'rt'))
    agents_by_subr: Dict[str, Dict[str, List[float]]] = json.load(gzip.open(agents_path, 'rt'))

    rows: List[Dict] = []
    for sr in sorted(set(auths_by_subr) & set(agents_by_subr)):
        bias_path = script_dir / f"data/bias/bias_values_{sr.lower()}.json"
        biased_authors = load_bias_authors(bias_path)

        if not biased_authors:
            rows.append({
                "subreddit": sr,
                "n_overlap": 0,
                "mape": None
            })
            continue

        common = [a for a in auths_by_subr[sr] if a in biased_authors and a in agents_by_subr[sr]]
        if len(common) < 2:
            rows.append({
                "subreddit": sr,
                "n_overlap": len(common),
                "mape": None
            })
            continue

        y_true = [itemgetter(1)(auths_by_subr[sr][a]) for a in common]
        y_pred = [itemgetter(1)(agents_by_subr[sr][a]) for a in common]
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rows.append({
            "subreddit": sr,
            "n_overlap": len(common),
            "mape": mape
        })

    out_path = script_dir / 'exit_order_mape_biased_only.csv'
    pd.DataFrame(rows).sort_values('subreddit').to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()


