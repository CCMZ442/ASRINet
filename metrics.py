"""Foreground semantic boundary counts; fixed 2% diagonal band."""
import numpy as np


def erode_square(mask, radius, outside=False):
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(radius):
        padded = np.pad(result, 1, constant_values=outside)
        result = np.logical_and.reduce([
            padded[y:y+result.shape[0], x:x+result.shape[1]]
            for y in range(3) for x in range(3)])
    return result


def boundary_counts(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.ndim != 2 or pred.shape != target.shape:
        raise ValueError('matching two-dimensional masks required')
    if not np.isin(pred, [0, 1]).all() or not np.isin(target, [0, 1, 255]).all():
        raise ValueError('unexpected label')
    radius = max(1, int(round(.02 * np.hypot(*pred.shape))))
    valid = erode_square(target != 255, radius, outside=True)
    p, g = pred == 1, target == 1
    pb, gb = p & ~erode_square(p, radius), g & ~erode_square(g, radius)
    return dict(boundary_intersection=int((pb & gb & valid).sum()),
                boundary_union=int(((pb | gb) & valid).sum()),
                boundary_valid_pixels=int(valid.sum()), boundary_radius=radius)


def summarize(rows, classes):
    def ratio(a, b):
        return a / b if b else None
    per_class = []
    for i, name in enumerate(classes):
        selected = [r for r in rows if r['class_index'] == i]
        totals = {k: sum(r[k] for r in selected) for k in
                  ('tp', 'fp', 'fn', 'tn', 'boundary_intersection', 'boundary_union')}
        t = totals
        per_class.append(dict(name=name, episodes=len(selected), counts=t,
                              iou=ratio(t['tp'], t['tp']+t['fp']+t['fn']),
                              precision=ratio(t['tp'], t['tp']+t['fp']),
                              recall=ratio(t['tp'], t['tp']+t['fn']),
                              boundary_iou=ratio(t['boundary_intersection'], t['boundary_union'])))
    totals = {k: sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn', 'tn')}
    t = totals
    def mean_present(k):
        vals = [c[k] for c in per_class]
        return float(np.mean(vals)) if all(v is not None for v in vals) else None
    return dict(per_class=per_class, class_mean_iou=mean_present('iou'),
                boundary_mean_iou=mean_present('boundary_iou'),
                fbiou=(ratio(t['tp'], t['tp']+t['fp']+t['fn']) +
                       ratio(t['tn'], t['tn']+t['fp']+t['fn'])) / 2,
                precision=ratio(t['tp'], t['tp']+t['fp']),
                recall=ratio(t['tp'], t['tp']+t['fn']))
