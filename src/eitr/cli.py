"""EITR commands; network calls require explicit stage flags and model settings."""
import argparse
import json
from pathlib import Path
import sys
from eicr.contracts import ContractError, digest, read_json, require, write_json
from eicr.packer import CharacterTokenizer, TiktokenTokenizer
from eicr.provider import ChatBackend
from .config import Config
from .native import canonicalize, import_pack
from .index import build_index, save_index, load_index
from .evidence import JsonTokenizer
from .pipeline import retrieve, compare
from .explorer import explore
from .evaluate import summarize
from .scheduler import ARMS


def parser():
    cli = argparse.ArgumentParser(prog="eitr")
    sub = cli.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import-mragent")
    imp.add_argument("--pack", required=True); imp.add_argument("--manifest", required=True); imp.add_argument("--out", required=True)
    canon = sub.add_parser("canonicalize")
    canon.add_argument("--snapshot", required=True); canon.add_argument("--out", required=True)
    build = sub.add_parser("build")
    build.add_argument("--snapshot", required=True); build.add_argument("--config")
    build.add_argument("--fact-vectors", help="Aligned JSON with snapshot_hash, encoder, fact_ids, vectors")
    build.add_argument("--out", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--index", required=True)
    for name in ("retrieve", "compare", "explore-control", "batch"):
        command = sub.add_parser(name)
        command.add_argument("--index", required=True)
        if name == "batch": command.add_argument("--questions", required=True, help="JSON list of unlabeled questions")
        else: command.add_argument("--question", required=True)
        command.add_argument("--config"); command.add_argument("--out", required=True)
        command.add_argument("--frontend", help="Pinned supplied frontend JSON; see documentation")
        command.add_argument("--tokenizer", required=True, help="characters (tests), json, or tiktoken encoding")
        command.add_argument("--tokenizer-file"); command.add_argument("--tokenizer-sha256")
        command.add_argument("--api-base"); command.add_argument("--model"); command.add_argument("--key-env", default="OPENAI_API_KEY")
        command.add_argument("--llm-frontend", action="store_true")
        command.add_argument("--generate", action="store_true")
        command.add_argument("--max-output-tokens", type=int, default=2048)
        if name == "retrieve": command.add_argument("--arm", choices=ARMS, default="P4")
        if name in ("compare", "batch"):
            command.add_argument("--arms", nargs="+", choices=ARMS, default=["P0", "P1", "P2", "P3", "P4"])
    evaluation = sub.add_parser("summarize")
    evaluation.add_argument("--rows", required=True); evaluation.add_argument("--out", required=True)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if getattr(args, "out", None):
            require(not Path(args.out).exists(), "Output exists; choose a new path")
        if args.command == "import-mragent":
            write_json(args.out, import_pack(read_json(args.pack), read_json(args.manifest)))
        elif args.command == "canonicalize":
            write_json(args.out, canonicalize(read_json(args.snapshot)))
        elif args.command == "build":
            config = Config.from_dict(read_json(args.config)) if args.config else Config()
            save_index(build_index(read_json(args.snapshot), config.top_k,
                                  read_json(args.fact_vectors) if args.fact_vectors else None), args.out)
        elif args.command == "audit":
            index = load_index(args.index)
            print(json.dumps({"index_hash": index.index_hash, "representation": index.representation,
                              "owner": index.snapshot["owner_id"], "snapshot": index.snapshot["snapshot_id"],
                              "report": index.report}, ensure_ascii=False, indent=2))
        elif args.command == "summarize":
            write_json(args.out, summarize(read_json(args.rows)))
        else:
            index = load_index(args.index)
            config = Config.from_dict(read_json(args.config)) if args.config else Config()
            if args.tokenizer == "characters":
                tokenizer = CharacterTokenizer()
            elif args.tokenizer == "json":
                require(args.tokenizer_file and args.tokenizer_sha256, "Pin tokenizer file and SHA256")
                tokenizer = JsonTokenizer(args.tokenizer_file, args.tokenizer_sha256)
            else:
                tokenizer = TiktokenTokenizer(args.tokenizer)
            require(not(args.llm_frontend and args.frontend), "Choose cached or live frontend")
            if args.llm_frontend or args.generate or args.command == "explore-control":
                require(args.api_base and args.model, "Explicit API base and model required")
                require(args.tokenizer != "characters", "Character counter cannot budget a live model")
            def backend():
                return ChatBackend(args.api_base, args.model, args.key_env, max_output_tokens=args.max_output_tokens, temperature=0)
            front = backend() if args.llm_frontend else None
            answer = backend() if args.generate else None
            mentions = None
            if args.frontend:
                require(args.command != "batch", "Batch mentions belong in each question row")
                supplied = read_json(args.frontend)
                require(supplied.get("question_hash") == digest(args.question)
                        and supplied.get("snapshot_hash") == index.snapshot["content_hash"], "Cached frontend scope mismatch")
                mentions = supplied["mentions"]
            if args.command == "retrieve":
                result = retrieve(index, args.question, tokenizer, config=config, arm=args.arm, mentions=mentions,
                                  frontend_backend=front, answer_backend=answer)
            elif args.command == "explore-control":
                result = explore(index, args.question, tokenizer, backend(), config=config, mentions=mentions,
                                 frontend_backend=front, answer_backend=answer)
            elif args.command == "compare":
                require(not args.generate, "compare is retrieval-only; use batch --generate for answer runs")
                result = compare(index, args.question, tokenizer, config=config, arms=args.arms, mentions=mentions,
                                 frontend_backend=front)
            else:
                from .batch import run_batch
                result = run_batch(index, read_json(args.questions), tokenizer, config=config, arms=args.arms,
                                   frontend_backend=front, answer_backend=answer, output_dir=args.out)
                return 0
            write_json(args.out, result)
        return 0
    except (ContractError, ValueError, OSError, KeyError, TypeError) as error:
        # Native inputs may contain private text; do not echo parser errors or bodies.
        print("eitr: " + (str(error) if isinstance(error, ContractError) else type(error).__name__), file=sys.stderr)
        return 2
