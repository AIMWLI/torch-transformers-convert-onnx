import torch

from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoTokenizer, AutoModel

# 模型路径
model_name = '/Users/songjin/IdeaProjects/torch-transformers-convert-onnx/bge-large-zh-v1.5'

# 加载模型和分词器
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 准备输入
dummy_input = tokenizer("Hello, my dog is cute", return_tensors="pt")

# 导出模型为 ONNX 格式
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    "model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input_ids', 'attention_mask'],
    output_names=['output'],
    dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("Model exported to model.onnx")


# 加载并优化模型
def optimize_onnx_model(onnx_model_path, optimized_model_path):
    # 使用 onnxruntime 加载和优化模型
    # opt_model = optimize_model(onnx_model_path, model_type='bert')
    # opt_model.save_model_to_file(optimized_model_path)
    print(f"Optimized model saved to {optimized_model_path}")

# 调用优化函数
# optimize_onnx_model("model.onnx", "../torch_export/onnx/model_optimized.onnx")
