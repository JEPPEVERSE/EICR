"""Load pinned upstream control flow without importing its credential-dependent app.

Upstream sources stay in the caller's checkout; nothing is vendored or renamed.
Only the actual native loop, tool bridge/schema, prompts and JSON parser are
compiled. The controlled memory/transport adapters are implemented separately.
"""
import ast
import copy
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from eicr.contracts import digest, require

FILES = ('llm/controller.py', 'agent/tools.py', 'agent/agent.py', 'prompts/prompts.py', 'common/utils.py')


def pin_runtime(root):
    root = Path(root)
    return {'schema': 'eitr-native-runtime-1', 'upstream': 'https://github.com/Ji-shuo/MRAgent',
            'files': {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in FILES},
            'policy': 'Execute pinned native tool loop and bridge with controlled raw-evidence tools; not M0.'}


def load_runtime(root, manifest):
    require(manifest == pin_runtime(root), 'MRAgent source pin mismatch; review and repin explicitly')
    trees = {f: ast.parse((Path(root)/f).read_text()) for f in FILES}
    logger = logging.Logger('eitr.native.private', level=logging.CRITICAL+1)
    ns = {'json': json, 'logger': logger, 'config': SimpleNamespace(MODEL='controlled', MAX_ROUNDS=8, MAX_TOOL_CALLS=32),
          'API_STATS': {'rounds': 0, 'tool_calls': 0}, 'MemoryController': object}
    def install(nodes, filename):
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + copy.deepcopy(nodes), type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(Path(root)/filename), 'exec'), ns)
    def named(file, name, kind):
        found = [n for n in trees[file].body if isinstance(n, kind) and n.name == name]
        require(len(found) == 1, 'Missing native definition: '+name)
        return found[0]
    install([named('prompts/prompts.py', 'Prompts', ast.ClassDef)], 'prompts/prompts.py')
    install([named('common/utils.py', 'extract_json_from_content', ast.FunctionDef)], 'common/utils.py')
    llm = named('llm/controller.py', 'LLM', ast.ClassDef)
    methods = [n for n in llm.body if isinstance(n, ast.FunctionDef) and n.name == 'chat_with_tools_once']
    require(len(methods) == 1, 'Native loop unavailable')
    install(methods, 'llm/controller.py')
    schema = [n for n in trees['agent/tools.py'].body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'TOOLS' for t in n.targets)]
    require(len(schema) == 1, 'Native tool schema unavailable')
    install(schema + [named('agent/tools.py', 'ToolBridge', ast.ClassDef)], 'agent/tools.py')
    return SimpleNamespace(loop=ns['chat_with_tools_once'], tools=ns['TOOLS'], bridge=ns['ToolBridge'],
                           prompts=ns['Prompts'], identity=digest(manifest), manifest=manifest)
