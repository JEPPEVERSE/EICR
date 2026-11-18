import pytest

from eicr.contracts import ContractError
from eicr.query_map import ground_mentions


def test_unique_literal_surface_grounds_without_model_counting():
    question = '我在 September 做 halloumi 沙拉。'
    original = [{'surface_text': 'halloumi', 'role': 'entry', 'span_start': 0, 'span_end': 8}]
    result = ground_mentions(question, original)
    assert question[result[0]['span_start']:result[0]['span_end']] == 'halloumi'
    assert original[0]['span_start'] == 0
    assert result[0]['role'] == 'entry'


def test_repeated_surface_needs_valid_occurrence():
    with pytest.raises(ContractError):
        ground_mentions('John met John', [{'surface_text': 'John', 'role': 'entry'}])
    assert ground_mentions('John met John', [{'surface_text': 'John', 'span_start': 9, 'span_end': 13}])[0]['span_start'] == 9


def test_grounding_never_repairs_semantic_or_lexical_text():
    for surface in ['Halloumi', 'salad', '', None]:
        with pytest.raises(ContractError):
            ground_mentions('halloumi', [{'surface_text': surface}])
