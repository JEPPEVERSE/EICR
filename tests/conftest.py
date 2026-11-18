import pytest


@pytest.fixture
def snapshot():
    # Synthetic memory only: three overlapping core labels and one separate cue.
    sources = [{"source_id": "s"+str(i), "text": text} for i, text in enumerate([
        "Halloumi salad used a pinch of smoked paprika.",
        "The salad was served on Sunday.",
        "A pinch of paprika was added to the dressing.",
        "The shopping trip bought Mediterranean blend and snacks.",
        "The museum visit was on Wednesday."
    ])]
    return {
        "owner_id": "synthetic", "snapshot_id": "synthetic-v1", "sources": sources,
        "cues": [{"cue_id": c, "text": c} for c in ("halloumi", "pinch", "paprika", "shopping", "empty", "owner")],
        "evidence": [{"evidence_id": "e"+str(i), "text": s["text"], "origins": [s["source_id"]], "topics": ["cooking"] if i < 2 else [], "tags": ["food"] if i < 4 else []} for i, s in enumerate(sources)],
        "edges": [{"cue_id": c, "evidence_id": "e"+str(i)} for c, ids in {
            "halloumi": [0, 1], "pinch": [0, 2], "paprika": [0, 1, 2], "shopping": [3], "owner": [0,1,2,3,4]
        }.items() for i in ids],
        "provenance": {"allowed_source_ids": [s["source_id"] for s in sources], "extraction_prefix_verified": True}
    }
