# MM-RAG
一个多模态RAG框架，赋予LLM输出图文交错结果的能力。

该框架目前仅支持PDF格式的知识库，解析后放在以下路径:
KB_PATH      = JSON格式的解析结果路径
IMAGE_ROOT   = 图像文件根目录

采用load_corpus_parallel函数处理知识库的解析结果, 
采用save_docs与load_serialized_docs保存和读取处理结果。

需要预先下载好稠密检索模型并放在在DENSE_MODEL路径下
