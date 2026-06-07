from opencompass.models import LLaDAModel

models = [
    dict(
        type=LLaDAModel,
        abbr='llada-1.5-8b-instruct',
        path='GSAI-ML/LLaDA-1.5',  # 服务器上改成本地路径，如 /home/ubuntu/.cache/huggingface/hub/models--GSAI-ML--LLaDA-1.5/snapshots/xxx
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
