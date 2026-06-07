from opencompass.models import LLaDAModelgtrfc

models = [
    dict(
        type=LLaDAModelgtrfc,
        abbr='llada-8b-instruct',
        path='/home/nvme04/worldf/wm_mem/model',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
