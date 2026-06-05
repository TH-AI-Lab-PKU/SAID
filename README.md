SAID: Accelerating Diffusion-Based Language Models via Scaffold-Aware Iterative Decoding

cd opencompass
pip install -e .

Follow the path in opencompass.configs.models.dllm.llada_instruct_8b to download the LLaDA-8B-Instruct model from Hugging Face.
#huggingface-cli download GSAI-ML/LLaDA-8B-Instruct


1. Run the original llada command:
python run.py examples/llada_instruct_gen_arcc_length512_block512.py
python run.py examples/llada_instruct_gen_gpqa_length64_block64_confidence.py
python run.py examples/llada_instruct_gen_gsm8k_length512_block512_confidence.py
python run.py examples/llada_instruct_gen_mbpp_length256_block256_confidence.py
python run.py examples/llada_instruct_gen_math_length512_block512_confidence.py
python run.py examples/llada_instruct_gen_mmlupro_length256_block256.py



2. Run our method:
python run.py examples/llada_instruct_gsm8k.py
python run.py examples/llada_instruct_math.py
python run.py examples/llada_instruct_gpqa.py
python run.py examples/llada_instruct_arcc.py
python run.py examples/llada_instruct_mmlupro.py

#For the MBPP dataset, you need to change CONF_THRESH in my_generate_gtr_fc.py to 0.7, and change the steps of num_transfer_hard to 8 before running:
python run.py examples/llada_instruct_mbpp.py
