import os
from pathlib import Path
from create_conf_bias import read_subreddits, ALL_SUBREDDITS

def main():
    # Ensure we run from repo root
    os.chdir(Path(__file__).resolve().parent)

    subs = ALL_SUBREDDITS
    subreddits = read_subreddits(subs)

    report = []
    for sr in subreddits:
        users_total = len(sr.members)
        users_2plus = sum(1 for r in sr.members.values() if len(r.comments_with_parents) >= 2)
        users_1 = sum(1 for r in sr.members.values() if len(r.comments_with_parents) == 1)
        users_0 = users_total - users_2plus - users_1
        report.append((sr.name, users_total, users_2plus, users_1, users_0))

    # Sort by users_2plus descending
    report.sort(key=lambda x: x[2], reverse=True)

    # Print header and rows
    print(f"{'subreddit':<22} {'users_total':>11} {'users_2+':>9} {'users_1':>8} {'users_0':>8}")
    for name, total, two_plus, one, zero in report:
        print(f"{name:<22} {total:>11} {two_plus:>9} {one:>8} {zero:>8}")

if __name__ == "__main__":
    main()