import pytest
from pydantic import ValidationError

from src.schemas import QueryRequest


def test_valid_query_request():
    req = QueryRequest(question="How do I endorse a check to a third party?")
    assert req.question == "How do I endorse a check to a third party?"
    assert req.use_cache is True

def test_query_request_strips_whitespace():
    req = QueryRequest(question="   What is an HSA?   ")
    assert req.question == "What is an HSA?"

def test_empty_query_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="")

def test_whitespace_only_query_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="      ")

def test_too_short_query_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="hi")  # < 3 chars

