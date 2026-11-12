import argparse
from pathlib import Path

from ..config import *
from ..infer import ChatModel
from ..eval.runner import Runner
from ..eval.math500 import *
from ..eval.mmlu import *

config = dict(CONFIG_QWEN06_BASE)

parser = argparse.ArgumentParser()
parser.add_argument('--bench', '--benchmark', choices=['math500', 'mmlu'], default='math500')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--template', choices=['cot', 'vanilla', 'answer-only'], default='cot')
parser.add_argument('--max-tokens', '--max-new-tokens', type=int, default=config['max_new_tokens'])
parser.add_argument('--temp', '--temperature', type=float, default=config['temperature'])
parser.add_argument('--p', '--nucleus-top-p', type=float, default=config['nucleus_top_p'])
parser.add_argument('--k', '--consistency-k', type=int, default=config['consistency_k'])
parser.add_argument("--task-range", nargs=2, type=int, default=None)
parser.add_argument("--output-dir", type=Path, default=None)
args = parser.parse_args()


# ===== construct

if args.bench == 'math500':
    templates = {'cot': MATH500_TEMP_COT, 'vanilla': MATH500_TEMP_VANILLA, 'answer-only': MATH500_TEMP_ANSWER_ONLY}
    tasks = Math500Manifest(template=templates[args.template], split=args.split)
elif args.bench == 'mmlu':
    templates = {'cot': MMLU_TEMP_COT, 'vanilla': MMLU_TEMP_VANILLA, 'answer-only': MMLU_TEMP_ANSWER_ONLY}
    tasks = MMLUManifest(template=templates[args.template], split=args.split)
else:
    raise ValueError

config['max_new_tokens'] = args.max_tokens
config['temperature'] = args.temp
config['nucleus_top_p'] = args.p
config['consistency_k'] = args.k
chatmodel = ChatModel(SPEC_QWEN06B, config)

tasks.load(indices=range(*args.task_range) if args.task_range else None, download=True)

out_path = (f'reasoninglab/out/records_{args.bench}_{args.template}.jsonl') if args.output_dir is None else args.output_dir
if out_path is not None: Path(out_path).parent.mkdir(parents=True, exist_ok=True)

evaluator = Runner(chatmodel, tasks=tasks, out_path=out_path)
evaluator.run()

"""
uv run -m reasoninglab.scripts.eval --bench math500 --template answer-only --max-tokens 5000 --temp 0.3 --p 0.85 --task-range 0 50
uv run -m reasoninglab.scripts.eval --bench mmlu --split dev --template vanilla --max-tokens 5000 --temp 0.3 --p 0.85 --task-range 0 50
"""
