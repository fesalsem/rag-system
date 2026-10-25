from evals.metrics.generation import aggregate, is_abstention


def test_is_abstention_matches_the_refusal_sentence():
    assert is_abstention("I could not find information about that in the uploaded documents.")
    assert is_abstention("The excerpts do not contain any information about that.")
    assert not is_abstention("The answer is on page 4.")


def test_aggregate_combines_faithfulness_and_abstention():
    rows = [
        {"type": "fact", "faithfulness": {"score": 1.0}},
        {"type": "fact", "faithfulness": {"score": 0.5}},
        {"type": "unanswerable", "abstained": True},
        {"type": "unanswerable", "abstained": False},
    ]
    result = aggregate(rows)
    assert result["faithfulness"] == 0.75
    assert result["abstention_accuracy"] == 0.5
    assert result["judged_answers"] == 2
