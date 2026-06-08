from opencompass.models import LLaDAModel

models = [
    dict(
        type=LLaDAModel,
        abbr='llada-8b-instruct',
        path='/home/ubuntu/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
