"""Keep source books, parallel sentence variants and translation directions together."""
from collections import Counter, defaultdict
import hashlib
import re
import unicodedata
from difflib import SequenceMatcher


def canonical(text):
    return "".join(c for c in unicodedata.normalize("NFC", text).casefold()
                   if unicodedata.category(c)[0] in "LMN" or c == "\\")


def grouped_split(records, train_ratio=0.9, seed=42):
    parents = list(range(len(records)))
    owners = {}
    near = defaultdict(list)

    def root(i):
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, row in enumerate(records):
        a, b = sorted(map(canonical, (row["input"], row["output"])))
        keys = [("pair", a, b)]
        if row["task_type"].startswith("translation"):
            keys.extend(("side", text) for text in (a, b) if text)
            # Include variant pairs whose two sides both contain small edits.
            english, target = ((row["input"], row["output"]) if "_en_to_" in row["task_type"]
                               else (row["output"], row["input"]))
            en, target = canonical(english), canonical(target)
            if min(len(en), len(target)) >= 24:
                for anchor in ((en[:12], target[:12]), (en[-12:], target[-12:])):
                    bucket = near[anchor]
                    for j, old_en, old_target in bucket:
                        if root(j) != root(i) and SequenceMatcher(None, en, old_en, autojunk=False).ratio() >= .94 and SequenceMatcher(None, target, old_target, autojunk=False).ratio() >= .94:
                            parents[root(i)] = root(j)
                    bucket.append((i, en, target))
        if re.fullmatch(r"M\d+", row.get("document", "")):
            keys.append(("book", row["document"]))
        for key in keys:
            parents[root(i)] = root(owners.setdefault(key, i))

    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[root(i)].append(row)
    train, validation = [], []
    book_groups = []
    for group in groups.values():
        if any(re.fullmatch(r"M\d+", row.get("document", "")) for row in group):
            book_groups.append(group)
            continue
        identifier = min(row["id"] for row in group)
        digest = hashlib.sha256(f"{seed}/{identifier}".encode()).digest()
        target = train if int.from_bytes(digest[:8], "big") / 2**64 < train_ratio else validation
        target.extend(group)
    # Whole-book holdouts can vary greatly in size. Select a small combination
    # that approximates the requested validation fraction for each book task.
    counts = [Counter(r["task_type"] for r in g) for g in book_groups]
    totals = sum(counts, Counter())
    def cost(held):
        return sum((held[task] / total - (1 - train_ratio)) ** 2 for task, total in totals.items())
    held = Counter()
    remaining = set(range(len(book_groups)))
    while remaining:
        best = min(remaining, key=lambda j: (cost(held + counts[j]), min(r["id"] for r in book_groups[j])))
        if cost(held + counts[best]) >= cost(held):
            break
        held.update(counts[best])
        validation.extend(book_groups[best])
        remaining.remove(best)
    for j in sorted(remaining):
        train.extend(book_groups[j])
    return train, validation
