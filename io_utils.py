"""File helpers for PACT-Surv experiments."""
import csv
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def export_predictions(path, d, ids, risk):
    csv_rows(path, [dict(patient=str(d['patients'][i]), time=float(d['time'][i]),
        event=int(d['event'][i]), risk=float(r)) for i, r in zip(ids, risk)])


def signature():
    return {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}
