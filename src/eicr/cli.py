"""Commands for immutable imports/indices and explicitly configured online runs."""

import argparse
import json
import sys
from pathlib import Path

from .config import IndexConfig
from .contracts import ContractError, read_json, write_json
from .diagnostics import diagnose
from .evaluate import summarize, validate_splits
from .import_mragent import import_pack
from .index import audit_index, build_index, load_index, save_index
from .packer import CharacterTokenizer, TiktokenTokenizer
from .pipeline import run
from .policy import choose_route, freeze_policy
from .provider import ChatBackend
from .query_map import map_query


def parser():
    cli = argparse.ArgumentParser(prog="eicr")
    sub = cli.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="Convert a frozen MRAgent pack with an explicit legal-prefix manifest")
    imp.add_argument("--pack", required=True); imp.add_argument("--manifest", required=True); imp.add_argument("--out", required=True)
    build = sub.add_parser("build", help="Build an immutable sparse index from a canonical snapshot")
    build.add_argument("--snapshot", required=True); build.add_argument("--config"); build.add_argument("--out", required=True)
    audit = sub.add_parser("audit-index"); audit.add_argument("--index", required=True)
    freeze = sub.add_parser("freeze-policy")
    freeze.add_argument("--index", required=True); freeze.add_argument("--selection", required=True); freeze.add_argument("--out", required=True)
    for name in ("diagnose", "run"):
        command = sub.add_parser(name)
        command.add_argument("--index", required=True); command.add_argument("--question", required=True)
        command.add_argument("--question-id"); command.add_argument("--frontend", help="JSON containing mentions, answer_need and query_shape")
        command.add_argument("--policy"); command.add_argument("--out", required=True)
        if name == "run":
            command.add_argument("--tokenizer", required=True, help="tiktoken encoding name, or characters for tests only")
            command.add_argument("--api-base"); command.add_argument("--model"); command.add_argument("--key-env", default="OPENAI_API_KEY")
            command.add_argument("--llm-frontend", action="store_true"); command.add_argument("--generate", action="store_true")
            command.add_argument("--dense", help="NPZ: chunk_vectors, query_vector; aligned with frozen chunk order")
            command.add_argument("--dense-manifest", help="JSON: NPZ hash, index, tokenizer, chunk order, question hash and model identity")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--rows", required=True); evaluate.add_argument("--out", required=True)
    splits = sub.add_parser("audit-splits")
    splits.add_argument("--rows", required=True)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if getattr(args, "out", None) and Path(args.out).exists():
            raise ContractError("Output already exists; choose a new path")
        if args.command == "import":
            write_json(args.out, import_pack(read_json(args.pack), read_json(args.manifest)))
        elif args.command == "build":
            config = IndexConfig.from_dict(read_json(args.config)) if args.config else IndexConfig()
            save_index(build_index(read_json(args.snapshot), config), args.out)
        elif args.command == "audit-index":
            print(json.dumps(audit_index(load_index(args.index)), ensure_ascii=False, indent=2))
        elif args.command == "freeze-policy":
            selection = read_json(args.selection)
            write_json(args.out, freeze_policy(load_index(args.index), selection["thresholds"], selection["selection_manifest"]))
        elif args.command in ("diagnose", "run"):
            index = load_index(args.index)
            front = read_json(args.frontend) if args.frontend else {}
            policy = read_json(args.policy) if args.policy else None
            options = {"question_id": args.question_id, "mentions": front.get("mentions"), "answer_need": front.get("answer_need", ""), "query_shape": front.get("query_shape", "unknown")}
            if args.command == "diagnose":
                query = map_query(index, args.question, **options)
                diagnostics = diagnose(index, query)
                write_json(args.out, {"query_map": query, "diagnostics": diagnostics, "route": choose_route(index, query, diagnostics, policy)})
            else:
                tokenizer = CharacterTokenizer() if args.tokenizer == "characters" else TiktokenTokenizer(args.tokenizer)
                frontend_backend = answer_backend = None
                if args.generate or args.llm_frontend:
                    if not args.api_base or not args.model:
                        raise ContractError("Explicit --api-base and --model required for network calls")
                    if args.llm_frontend:
                        frontend_backend = ChatBackend(args.api_base, args.model, args.key_env)
                    if args.generate:
                        answer_backend = ChatBackend(args.api_base, args.model, args.key_env)
                dense = {}
                if args.dense:
                    if not args.dense_manifest:
                        raise ContractError("--dense requires --dense-manifest")
                    dense_manifest = read_json(args.dense_manifest)
                    from hashlib import sha256
                    if sha256(Path(args.dense).read_bytes()).hexdigest() != dense_manifest.get("npz_sha256"):
                        raise ContractError("Dense NPZ hash mismatch")
                    import numpy as np
                    with np.load(args.dense, allow_pickle=False) as values:
                        dense = {"dense_vectors": values["chunk_vectors"], "query_vector": values["query_vector"],
                                 "dense_identity": dense_manifest["dense_identity"], "dense_manifest": dense_manifest}
                write_json(args.out, run(index, args.question, tokenizer, **options, policy=policy,
                                         frontend_backend=frontend_backend, answer_backend=answer_backend, **dense))
        elif args.command == "evaluate":
            write_json(args.out, summarize(read_json(args.rows)))
        elif args.command == "audit-splits":
            print(json.dumps(validate_splits(read_json(args.rows))))
        return 0
    except (ContractError, OSError, TypeError, KeyError) as error:
        print("eicr: " + str(error), file=sys.stderr)
        return 2
