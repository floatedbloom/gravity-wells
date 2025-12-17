import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_texts(path: Path, limit: int = 300) -> Tuple[List[str], List[str], List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    texts: List[str] = []
    ids: List[str] = []
    links: List[str] = []
    for item in data[:limit]:
        t = item.get("text") or item.get("body") or ""
        texts.append(str(t))
        ids.append(str(item.get("id")))
        links.append(str(item.get("permalink", "")))
    return texts, ids, links


def pick_pairs(sim: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    # Zero diagonal and flatten the upper triangle for ranking
    n = sim.shape[0]
    np.fill_diagonal(sim, 0.0)
    iu = np.triu_indices(n, k=1)
    sims = sim[iu]
    # High alignment: max similarity
    high_idx = int(np.argmax(sims))
    hi = iu[0][high_idx], iu[1][high_idx]
    # Low alignment: min similarity (likely 0)
    low_idx = int(np.argmin(sims))
    lo = iu[0][low_idx], iu[1][low_idx]
    # Neutral alignment: pick around median of nonzero sims or 40-60th percentile
    nonzero_mask = sims > 0
    if np.any(nonzero_mask):
        nz = sims[nonzero_mask]
        target = np.percentile(nz, 50)
        neutral_idx = int(np.argmin(np.abs(sims - target)))
    else:
        neutral_idx = len(sims) // 2
    ne = iu[0][neutral_idx], iu[1][neutral_idx]
    return hi, ne, lo


def main() -> None:
    root = Path(__file__).resolve().parent
    agree_path = root / "data" / "zipped" / "AmericanPolitics_2019-01_agree.json"
    texts, ids, links = load_texts(agree_path, limit=300)

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, max_features=20000)
    X = vectorizer.fit_transform(texts)
    sim = cosine_similarity(X)

    hi, ne, lo = pick_pairs(sim.copy())

    def fmt_pair(a: int, b: int, label: str) -> str:
        s = float(sim[a, b])
        ta = texts[a].replace("\n", " ")
        tb = texts[b].replace("\n", " ")
        ea = (ta[:240] + ("..." if len(ta) > 240 else ""))
        eb = (tb[:240] + ("..." if len(tb) > 240 else ""))
        return (
            f"{label} alignment | score={s:.3f}\n"
            f"A: id={ids[a]} link={links[a]}\n   {ea}\n"
            f"B: id={ids[b]} link={links[b]}\n   {eb}\n"
        )

    print(fmt_pair(hi[0], hi[1], "+"))
    print(fmt_pair(ne[0], ne[1], "neutral"))
    print(fmt_pair(lo[0], lo[1], "-"))


if __name__ == "__main__":
    main()


