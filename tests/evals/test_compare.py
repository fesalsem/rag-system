from evals.compare import compare


def test_a_drop_inside_the_threshold_passes():
    baseline = {"summary": {"recall@4": 0.80, "mrr": 0.70}}
    current = {"summary": {"recall@4": 0.79, "mrr": 0.69}}
    thresholds = {"recall@4": -0.02, "mrr": -0.02}
    assert compare(baseline, current, thresholds) == []


def test_a_drop_past_the_threshold_fails():
    baseline = {"summary": {"recall@4": 0.80}}
    current = {"summary": {"recall@4": 0.75}}
    failures = compare(baseline, current, {"recall@4": -0.02})
    assert len(failures) == 1
    assert "recall@4" in failures[0]


def test_a_missing_metric_is_skipped():
    assert compare({"summary": {}}, {"summary": {}}, {"recall@4": -0.02}) == []
