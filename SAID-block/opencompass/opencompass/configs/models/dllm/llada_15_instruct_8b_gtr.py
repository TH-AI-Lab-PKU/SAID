from opencompass.models import LLaDAGTRModel

models = [
    dict(
        type=LLaDAGTRModel,
        abbr='llada-1.5-8b-instruct-gtr',
        path='GSAI-ML/LLaDA-1.5',
        max_out_len=1024,
        batch_size=1,
        num_stages=3,
        rec_steps=2,
        diff_confidence_eos_eot_inf=True,
        diff_logits_eos_inf=False,
        run_cfg=dict(num_gpus=1),
    )
]
