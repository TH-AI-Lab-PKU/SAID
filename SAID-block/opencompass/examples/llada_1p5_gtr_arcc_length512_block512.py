from mmengine.config import read_base
with read_base():
    from opencompass.configs.datasets.ARC_c.ARC_c_gen import \
        ARC_c_datasets
    from opencompass.configs.models.dllm.llada_15_instruct_8b_gtr import \
        models as llada_gtr_models
datasets = ARC_c_datasets
models = llada_gtr_models
# Baseline: gen_steps=512, block512 → GTR: gen_steps=128, same block
eval_cfg = {'gen_blocksize': 512, 'gen_length': 512, 'gen_steps': 128, 'batch_size': 1, 'batch_size_': 1}
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
