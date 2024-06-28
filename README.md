![image](https://github.com/AIMWLI/torch-transformers-convert-onnx/assets/31265254/3e675441-258a-45e0-b067-31141d2f5a94)

# 第一种方法
参考: https://docs.spring.io/spring-ai/reference/api/embeddings/onnx.html
</br>
从远程down模型并转换成onnx
optimum-cli export onnx -m  BAAI/bge-large-zh-v1.5 onnx-output-folder

# 第二种
先下载模型到本地</br>
抱抱脸连接: https://huggingface.co/BAAI/bge-large-zh-v1.5/tree/main</br>
强内镜像地址:https://hf-mirror.com/</br>
然后用torch或transformer转换onnx

# 后续
玩法一: 用java操作onnx embedding后入库(milvus or 其他)</br>
玩法二: 用vllm或ollama加载模型生成接口, 实现增量数据同步