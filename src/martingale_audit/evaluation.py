"""Agreement with provisional labels, and message provenance checks.

Incomplete reference labels cannot measure extraction precision or specificity.
"""
from collections import Counter
from math import isfinite


def _index(observations):
    index = {}
    for o in observations:
        if not o.msg_id:
            raise ValueError("Comparison requires message IDs")
        key = (o.msg_id, o.claim)
        if key in index:
            raise ValueError(f"Duplicate message/claim pair: {key}")
        if not (isfinite(o.belief) and 0 <= o.belief <= 1
                and isfinite(o.evidence) and 0 <= o.evidence <= 1):
            raise ValueError("Labels must be finite and in [0, 1]")
        index[key] = o
    return index


def compare_labels(reference, predicted):
    ref, pred = _index(reference), _index(predicted)
    matched = sorted(ref.keys() & pred.keys())
    n = len(matched)
    scale = (0.1, 0.3, 0.5, 0.7, 0.9)
    ordinal = lambda p: min(range(5), key=lambda i: abs(scale[i] - p))
    evidence = Counter()
    disagreements = []
    for key in matched:
        a, b = ref[key], pred[key]
        evidence[f"reference_{int(a.evidence >= .5)}_predicted_{int(b.evidence >= .5)}"] += 1
        if ordinal(a.belief) != ordinal(b.belief) or (a.evidence >= .5) != (b.evidence >= .5):
            disagreements.append({"msg_id": key[0], "claim": key[1],
                                  "reference_belief": a.belief, "predicted_belief": b.belief,
                                  "reference_evidence": a.evidence, "predicted_evidence": b.evidence})
    return {
        "reference_pairs": len(ref), "predicted_pairs": len(pred), "matched_pairs": n,
        "coverage": n / len(ref) if ref else None,
        "missing_reference_pairs": [{"msg_id": k[0], "claim": k[1]} for k in sorted(ref.keys() - pred.keys())],
        "unlabelled_prediction_pairs": len(pred.keys() - ref.keys()),
        "belief_mae_on_matches": sum(abs(ref[k].belief - pred[k].belief) for k in matched) / n if n else None,
        "ordinal_agreement_on_matches": sum(ordinal(ref[k].belief) == ordinal(pred[k].belief) for k in matched) / n if n else None,
        "evidence_agreement_on_matches": sum((ref[k].evidence >= .5) == (pred[k].evidence >= .5) for k in matched) / n if n else None,
        "evidence_confusion_on_matches": dict(evidence), "disagreements": disagreements,
        "scope": "Agreement with one provisional annotator on covered pairs. Missing labels are not negatives. These metrics do not establish IID noise, truth accuracy, or valid e-values.",
    }


def check_provenance(observations, messages, timestamp_tolerance=.001):
    messages, observations = list(messages), list(observations)
    index = {m.id: m for m in messages}
    findings = []
    duplicates = [mid for mid, n in Counter(m.id for m in messages).items() if n > 1]
    for o in observations:
        m = index.get(o.msg_id)
        base = {"msg_id": o.msg_id, "claim": o.claim, "agent": o.agent}
        if m is None:
            findings.append({**base, "code": "missing_message"})
        elif o.agent != m.agent:
            findings.append({**base, "code": "author_mismatch", "source_author": m.agent})
        elif not isfinite(o.ts) or abs(o.ts - m.ts) > timestamp_tolerance:
            findings.append({**base, "code": "timestamp_mismatch", "source_ts": m.ts, "label_ts": o.ts})
    return {"source_messages": len(messages), "observations": len(observations),
            "matched_message_ids": sum(o.msg_id in index for o in observations),
            "duplicate_source_ids": duplicates, "timestamp_tolerance_seconds": timestamp_tolerance,
            "passed": not findings and not duplicates, "findings": findings,
            "scope": "Checks source identity, author, and timestamp only; does not verify label semantics."}
