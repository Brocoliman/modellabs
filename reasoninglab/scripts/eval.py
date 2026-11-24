import argparse
from dataclasses import replace
from pathlib import Path

from ..config import *
from ..infer import ChatModel
from ..eval.runner import Runner
from ..eval.math500 import *
from ..eval.mmlu import *

parser = argparse.ArgumentParser()
parser.add_argument('--bench', '--benchmark', choices=['math500', 'mmlu'], default='math500')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--template', choices=['cot', 'vanilla', 'answer-only'], default='cot')
parser.add_argument('--max-tokens', '--max-new-tokens', type=int, default=SAMPLER_QWEN06.max_new_tokens)
parser.add_argument('--temp', '--temperature', type=float, default=SAMPLER_QWEN06.temperature)
parser.add_argument('--p', '--nucleus-top-p', type=float, default=SAMPLER_QWEN06.nucleus_top_p)
parser.add_argument('--k', '--consistency-k', type=int, default=EVALUATOR_QWEN06.consistency_k)
parser.add_argument("--task-range", nargs=2, type=int, default=EVALUATOR_QWEN06.task_range)
parser.add_argument("--output-dir", type=Path, default=EVALUATOR_QWEN06.out_path)
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

sampling = replace(SAMPLER_QWEN06, **{k: v for k, v in
        dict(temperature=args.temp, nucleus_top_p=args.p,
                max_new_tokens=args.max_tokens).items() if v is not None})
chatmodel = ChatModel(SPEC_QWEN06, sampling, QWEN06_BASE)

out_path = (f'reasoninglab/out/records_{args.bench}_{args.template}.jsonl') if args.output_dir is None else args.output_dir
if out_path is not None: Path(out_path).parent.mkdir(parents=True, exist_ok=True)
evaluate = replace(EVALUATOR_QWEN06, **{k: v for k, v in
        dict(consistency_k=args.k, task_range=args.task_range,
                out_path=out_path).items() if v is not None})

runner = Runner(chatmodel, tasks=tasks, evaluate=evaluate)
runner.run()


