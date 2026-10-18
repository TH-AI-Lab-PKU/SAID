from opencompass.models import LLaDASAIDModel

models = [
    dict(
        type=LLaDASAIDModel,
        abbr='llada-1.5-8b-instruct-said',
        path='/data/SAID/model',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]