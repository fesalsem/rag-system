import json

import pytest

from evals.judge import DeepSeekJudge, JudgeConfig, JudgeError


def response(claims):
    return {
        "model": "deepseek-chat-2026-10",
        "choices": [
            {"message": {"content": json.dumps({"claims": claims})}}
        ],
    }


def judge(tmp_path, fake_response):
    calls = []

    def transport(payload):
        calls.append(payload)
        return fake_response

    return (
        DeepSeekJudge(
            JudgeConfig(api_key="test-key"),
            cache_dir=tmp_path,
            transport=transport,
        ),
        calls,
    )


def test_faithfulness_is_the_supported_fraction(tmp_path):
    subject, _ = judge(
        tmp_path,
        response(
            [
                {"claim": "A", "supported": True, "evidence": "ctx"},
                {"claim": "B", "supported": False, "evidence": ""},
            ]
        ),
    )
    result = subject.score_faithfulness("q", "a", "ctx")
    assert result["total"] == 2
    assert result["supported"] == 1
    assert result["score"] == 0.5
    assert result["model"] == "deepseek-chat-2026-10"


def test_a_claimless_answer_scores_one(tmp_path):
    subject, _ = judge(tmp_path, response([]))
    assert subject.score_faithfulness("q", "a", "ctx")["score"] == 1.0


def test_cache_prevents_a_second_call(tmp_path):
    subject, calls = judge(
        tmp_path, response([{"claim": "A", "supported": True, "evidence": "ctx"}])
    )
    subject.score_faithfulness("q", "a", "ctx")
    subject.score_faithfulness("q", "a", "ctx")
    assert len(calls) == 1


def test_unparseable_output_raises(tmp_path):
    subject, _ = judge(
        tmp_path,
        {"choices": [{"message": {"content": "not json"}}]},
    )
    with pytest.raises(JudgeError):
        subject.score_faithfulness("q", "a", "ctx")


def test_request_is_deterministic_and_json_mode(tmp_path):
    subject, calls = judge(tmp_path, response([]))
    subject.score_faithfulness("q", "a", "ctx")
    payload = calls[0]
    assert payload["temperature"] == 0
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["model"] == "deepseek-chat"
