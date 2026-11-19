"""One shared grounded frontend, with fair seed groups and native scope."""
from collections import defaultdict
import json
from eicr.contracts import ContractError, digest, lookup, require
from eicr.provider import FRONTEND_PROMPT, ProviderTransportError
from eicr.query_map import ground_mentions, map_query
from eicr.retrieval import SharedRetriever


def mapping_config_hash(config):
    return digest({k:getattr(config,k) for k in ("max_query_cues","max_keys_per_cue","max_tags_per_key","max_seed_tags")})


def map_groups(index, question, config, mentions=None, backend=None):
    require(isinstance(question, str) and question.strip(), "Empty question")
    require(not (backend is not None and mentions is not None), "Choose supplied or LLM frontend")
    start = len(backend.calls) if backend is not None else 0
    errors, attempts = [], []
    if backend is not None:
        for attempt in range(2):
            try:
                raw = backend.chat_json(FRONTEND_PROMPT, json.dumps({"question": question}, ensure_ascii=False))
                attempts.append(raw)
                require(isinstance(raw, dict), "Frontend response must be an object")
                grounded = ground_mentions(question, raw.get("mentions"))
                query = map_query(index, question, grounded)
                query["frontend_basis"] = "llm_literal_spans"
                break
            except ProviderTransportError as error:
                errors.append(type(error).__name__)
                query = map_query(index, question, [])
                break
            except (ContractError, TypeError, KeyError) as error:
                errors.append(type(error).__name__)
        else:
            query = map_query(index, question, [])
    else:
        query = map_query(index, question, ground_mentions(question, mentions) if mentions is not None else None)
    links = defaultdict(lambda: defaultdict(set))
    for link in index.snapshot["links"]:
        links[link["cue_id"]][link["tag_id"]].add(link["fact_id"])
    tags = index.snapshot["tags"]
    ranker = SharedRetriever([{"span_id": t["tag_id"], "text": t["text"]} for t in tags])
    scores = dict(zip([t["tag_id"] for t in tags], ranker.lexical_scores(question)))
    groups, seen, dropped = [], {}, []
    for mention in query["mentions"]:
        if mention["role"] == "scope":
            continue
        name = lookup(mention["surface_text"])
        keys = mention["candidate_cue_ids"][:config.max_keys_per_cue] if mention["role"] == "entry" else []
        # Only a unique exact/attested-alias mapping establishes identity. Two
        # fuzzy observations sharing candidates must remain separate observations.
        verified = len(keys) == 1 and mention["status"] == "unique" and mention["match_method"] in ("exact", "explicit_alias")
        identity = ("verified_key", keys[0]) if verified else ("surface", name)
        if identity in seen:
            group = seen[identity]
            group["mentions"].append(dict(mention))
            group["surfaces"] = sorted(set(group["surfaces"] + [mention["surface_text"]]), key=lambda s: (lookup(s), s))
            group["text"] = " ".join(group["surfaces"])
            continue
        if len(groups) >= config.max_query_cues:
            dropped.append(name)
            continue
        candidates = defaultdict(set)
        tag_truncations = []
        for key in keys:
            ordered = sorted(links[key], key=lambda t: (-scores[t], t))
            tids = ordered[:config.max_tags_per_key]
            if len(ordered) > len(tids):
                tag_truncations.append({"key_id": key, "reason": "per_key_tag_cap", "omitted_tags": ordered[len(tids):]})
            for tid in tids:
                candidates[tid].update(links[key][tid])
        order = sorted(candidates, key=lambda t: (-scores[t], t))
        group = {"group_id": digest(identity)[:16], "text": mention["surface_text"],
                       "identity": list(identity), "surfaces": [mention["surface_text"]], "mentions": [dict(mention)],
                       "key_candidates_truncated": mention["role"] == "entry" and (mention.get("candidate_cues_truncated", False) or len(mention["candidate_cue_ids"]) > len(keys)),
                       "tag_truncations": tag_truncations,
                       "key_ids": keys, "candidate_tags": order, "seed_tags": [],
                       "facts_by_tag": {t: sorted(candidates[t]) for t in order},
                       "match_method": mention["match_method"]}
        groups.append(group)
        seen[identity] = group
    admitted = set()
    for pos in range(max((len(g["candidate_tags"]) for g in groups), default=0)):
        for g in groups:
            if pos >= len(g["candidate_tags"]):
                continue
            t = g["candidate_tags"][pos]
            if t in admitted or len(admitted) < config.max_seed_tags:
                admitted.add(t); g["seed_tags"].append(t)
    active = [g for g in groups if g["seed_tags"]]
    for g in groups:
        g["support_weight"] = 1 / len(active) if g["seed_tags"] else 0.
        g["truncated_tags"] = [t for t in g["candidate_tags"] if t not in g["seed_tags"]]
        g["facts_by_tag"] = {t: g["facts_by_tag"][t] for t in g["seed_tags"]}
    return {"schema": "eitr-query-1", "index_hash": index.index_hash, "question": question,
            "mapping_config_hash": mapping_config_hash(config),
            "groups": groups, "query_map": query, "dropped_cues": dropped,
            "frontend_errors": errors, "frontend_outputs": attempts,
            "frontend_calls": list(backend.calls[start:]) if backend is not None else [],
            "frontend_request": {"question_hash": digest(question), "prompt_hash": digest(FRONTEND_PROMPT)}}
