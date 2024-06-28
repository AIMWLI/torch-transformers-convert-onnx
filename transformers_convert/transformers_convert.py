from pathlib import Path

import torch
from transformers.convert_graph_to_onnx import convert

filename = r'/Users/songjin/IdeaProjects/torch-transformers-convert-onnx/bge-large-zh-v1.5'
print(torch.onnx.symbolic_opset18)
convert(framework="pt", model=filename, tokenizer=filename, pipeline_name="feature-extraction", opset=17, output=Path(
    "../transformers_convert/onnx/bge-large-zh-v1.5-normalized.onnx"))
