import pytest
from eitr.native import canonicalize


@pytest.fixture
def raw_memory():
    return {
        "owner_id": "owner-A", "snapshot_id": "frozen-1",
        "provenance": {"extraction_prefix_verified": True, "allowed_source_ids": ["D1:1", "D1:2", "D1:10", "D2:1"]},
        "sources": [
            {"source_id": "D1:1", "session_id": "D1", "text": "I bought paprika and herbs at the shop.",
             "speaker": "A", "session_datetime": "2024-09-15", "datetime_provenance": "native_verified"},
            {"source_id": "D1:2", "session_id": "D1", "text": "For halloumi salad I used a pinch of herbs.",
             "speaker": "A", "session_datetime": "2024-09-15", "datetime_provenance": "native_verified"},
            {"source_id": "D1:10", "session_id": "D1", "text": "The shop also sold popcorn.", "speaker": "A"},
            {"source_id": "D2:1", "session_id": "D2", "text": "The train to London was late.", "speaker": "A"}],
        "cues": [{"cue_id": c, "text": c, "aliases": []} for c in ("paprika", "herbs", "halloumi", "shop", "train", "I")],
        "evidence": [
            {"evidence_id": "f1", "text": "Bought spices", "origins": ["D1:1"], "tags": ["shopping"]},
            {"evidence_id": "f2", "text": "Used herbs", "origins": ["D1:2"], "tags": ["cooking"]},
            {"evidence_id": "f3", "text": "Bought snacks", "origins": ["D1:10"], "tags": ["snacks"]},
            {"evidence_id": "f4", "text": "Train late", "origins": ["D2:1"], "tags": ["travel"]}],
        "edges": [{"cue_id": c, "evidence_id": f, "edge_origin": "native_keyword"}
                  for f, cs in (("f1", ["paprika", "herbs", "shop"]), ("f2", ["herbs", "halloumi"]),
                                ("f3", ["shop"]), ("f4", ["train"])) for c in cs]
                  + [{"cue_id": "I", "evidence_id": f, "edge_origin": "injected_speaker"} for f in ("f1", "f2", "f3", "f4")]
    }


@pytest.fixture
def snapshot(raw_memory):
    return canonicalize(raw_memory)
