"""预测路径：仅读取本轮冻结参数和预测CSV；不导入规划或任何X1程序。"""
import csv
import json
from common import OUT, METHODS, TASKS, DIMS

def load():
    with (OUT / 'predictions.csv').open() as f:
        rows = list(csv.DictReader(f))
    pred = {}
    for row in rows:
        key = (int(row['d']), int(row['ir']), row['method'], row['task'])
        assert key not in pred
        pred[key] = {k: float(row[k]) for k in ['recall_limit', 'recall_first_order', 'mu', 'm', 'w', 'q', 'B_recall']}
    assert len(pred) == 90
    assert set(pred) == {(d, ir, m, task) for d in DIMS for ir in range(3) for m in METHODS for task in TASKS}
    params = json.loads((OUT / 'parameters.json').read_text())
    params = {(int(p['ir']), p['method']): p for p in params}
    assert len(params) == 15
    return pred, params
