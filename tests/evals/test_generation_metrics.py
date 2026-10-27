from evals.metrics.generation import aggregate, has_refusal, is_abstention


def test_is_abstention_matches_the_refusal_sentence():
    assert is_abstention("I could not find information about that in the uploaded documents.")
    assert is_abstention("The excerpts do not contain any information about that.")
    assert not is_abstention("The answer is on page 4.")


def test_is_abstention_ignores_the_citation_appended_to_a_refusal():
    assert is_abstention(
        "I could not find information about that in the uploaded documents. "
        "[berkshire-2023-ar.pdf — Page 12]"
    )


def test_partial_answer_is_not_an_abstention():
    answer = (
        "RAG reached 70% accuracy. "
        "I could not find information about the vector count in the uploaded documents."
    )
    assert not is_abstention(answer)
    assert has_refusal(answer)


def test_full_answer_is_neither_a_refusal_nor_an_abstention():
    assert not is_abstention("The base Transformer uses 8 heads.")
    assert not has_refusal("The base Transformer uses 8 heads.")


def test_empty_answer_is_not_an_abstention():
    assert not is_abstention(None)
    assert not is_abstention("   ")


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


def test_aggregate_reports_refusals_on_answerable_questions():
    rows = [
        {"type": "fact", "abstained": False},
        {"type": "fact", "abstained": True},
        {"type": "fact", "abstained": False, "partial": True},
        {"type": "unanswerable", "abstained": True},
    ]
    result = aggregate(rows)
    assert result["false_abstention_rate"] == 0.3333
    assert result["partial_answers"] == 1
