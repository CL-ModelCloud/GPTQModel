# from transformers import AutoTokenizer, TextGenerationPipeline
# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
# import logging
# from datasets import load_dataset
#
# logging.basicConfig(
#     format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
# )
#
# pretrained_model_dir = "/monster/data/model/QwQ-32B-Preview"
# quantized_model_dir = "QwQ-32B-Preview-Quant"
#
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, trust_remote_code=False)
#
# def load_c4_dataset(tokenizer):
#     traindata = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train")
#     datas = []
#     for index, sample in enumerate(traindata):
#         tokenized = tokenizer(sample['text'])
#         if len(tokenized.data['input_ids']) < 2048:
#             datas.append(tokenized)
#             if len(datas) >= 256:
#                 break
#
#     return datas
#
# datas = load_c4_dataset(tokenizer)
#
# quantize_config = BaseQuantizeConfig(
#     bits=4,  # quantize model to 4-bit
#     group_size=128,  # it is recommended to set the value to 128
#     desc_act=True,  # set to False can significantly speed up inference but the perplexity may slightly bad
#     sym=True,
# )
#
# # load un-quantized model, by default, the model will always be loaded into CPU memory
# model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
#
# # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
# model.quantize(datas, batch_size=4)
#
# # save quantized model using safetensors
# # model.save_quantized(quantized_model_dir, use_safetensors=True)

from transformers import AutoTokenizer, TextGenerationPipeline
from gptqmodel import GPTQModel, QuantizeConfig, BACKEND
from gptqmodel.quantization import FORMAT
import logging
from datasets import load_dataset

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = "/monster/data/model/QwQ-32B-Preview"
quantized_model_dir = "QwQ-32B-Preview-Quant"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, trust_remote_code=False)

def load_c4_dataset(tokenizer):
    traindata = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train")
    datas = []
    for index, sample in enumerate(traindata):
        tokenized = tokenizer(sample['text'])
        if len(tokenized.data['input_ids']) < 2048:
            datas.append(tokenized)
            if len(datas) >= 256:
                break

    return datas

datas = load_c4_dataset(tokenizer)

quantize_config = QuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=True,  # set to False can significantly speed up inference but the perplexity may slightly bad
    sym=True,
    format=FORMAT.GPTQ
)

model = GPTQModel.load(pretrained_model_dir, quantize_config, trust_remote_code=False, torch_dtype='auto', backend=BACKEND.AUTO, device_map='auto')

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(datas, batch_size=4)

# save quantized model using safetensors
# model.save_quantized(quantized_model_dir, use_safetensors=True)