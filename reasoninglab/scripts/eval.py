import argparse
from pathlib import Path

from ..config import *
from ..infer import ChatModel
from ..eval.runner import Runner
from ..eval.math500 import *
from ..eval.mmlu import *

config = CONFIG_QWEN06_BASE

parser = argparse.ArgumentParser()
parser.add_argument('--bench', '--benchmark', choices=['math500', 'mmlu'], default='math500')
parser.add_argument('--template', choices=['cot', 'vanilla', 'answer-only'], default='cot')
parser.add_argument('--max-tokens', '--max-new-tokens', type=int, default=config['max_new_tokens'])
parser.add_argument('--temp', '--temperature', type=float, default=config['temperature'])
parser.add_argument('--p', '--nucleus-top-p', type=float, default=config['nucleus_top_p'])
parser.add_argument("--task-range", nargs=2, type=int, default=None)
parser.add_argument("--output-dir", type=Path, default='reasoninglab/out/records_math500.jsonl')
args = parser.parse_args()


# ===== construct

if args.bench == 'math500':
    templates = {'cot': MATH500_TEMP_COT, 'vanilla': MATH500_TEMP_VANILLA, 'answer-only': MATH500_TEMP_ANSWER_ONLY}
    tasks = Math500Manifest(template=templates[args.template])
elif args.bench == 'mmlu':
    templates = {'cot': MMLU_TEMP_COT, 'vanilla': MMLU_TEMP_VANILLA, 'answer-only': MMLU_TEMP_ANSWER_ONLY}
    tasks = MMLUManifest(template=templates[args.template])
else:
    raise ValueError

config['max_new_tokens'] = args.max_tokens
config['temperature'] = args.temp
config['nucleus_top_p'] = args.p
chatmodel = ChatModel(SPEC_QWEN06B, config)

tasks.load(range(*args.task_range))

evaluator = Runner(chatmodel, tasks=tasks, out_path=args.output_dir)
evaluator.run()

"""
uv run -m reasoninglab.scripts.eval --bench math500 --template answer-only --max-tokens 5000 --temp 0.3 --p 0.85 --task-range 0 100
uv run -m reasoninglab.scripts.eval --bench mmlu --template vanilla --max-tokens 5000 --temp 0.3 --p 0.85 --task-range 0 100
uv run -m reasoninglab.scripts.eval --bench mmlu --template cot --max-tokens 5000 --temp 0.3 --p 0.85 --task-range 0 100
"""
