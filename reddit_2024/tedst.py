import gzip, json
with gzip.open('../auths_by_subr-sorted.json.gz', 'rt') as f:
    data = json.load(f)
print(list(data.keys()))