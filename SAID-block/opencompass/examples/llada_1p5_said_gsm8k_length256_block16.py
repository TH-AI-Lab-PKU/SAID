from mmengine.config import read_base
with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import \
        gsm8k_datasets
    from opencompass.configs.models.dllm.llada_15_instruct_8b_said import \
        models as llada_said_models
datasets = gsm8k_datasets
models = llada_said_models
# Baseline: gen_steps=256, block16 → SAID: gen_steps=64, same block
eval_cfg = {'gen_blocksize': 16, 'gen_length': 256, 'gen_steps': 64, 'batch_size': 1, 'batch_size_': 1}
for model in models:
    model.update(eval_cfg)
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8,
        num_split=None,
        min_task_size=16,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=5
    ),
)
