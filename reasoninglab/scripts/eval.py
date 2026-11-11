import argparse

from ..config import *
from ..infer import ChatModel
from ..eval.runner import Runner
from ..eval.math500 import Math500Manifest
from ..eval.mmlu import MMLUManifest

#MMLUManifest.download()

Math500Manifest.download()
config = CONFIG_QWEN06_BASE
config['temperature'] = 0.3
config['nucleus_top_p'] = 0.9
chatmodel = ChatModel(SPEC_QWEN06B, config)

tasks = Math500Manifest()
tasks.load(range(10))

evaluator = Runner(chatmodel, 
        tasks=tasks, out_path='reasoninglab/out/records_math500.jsonl')
evaluator.run()