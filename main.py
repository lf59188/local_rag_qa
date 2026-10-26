# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import os, sys, json, time, asyncio, logging
from pathlib import Path
from typing import Sequence

# ---- 日志配置 ----
logging.basicConfig( level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)] )
logger = logging.getLogger("AgenticRAG")

EMBEDDING_MODEL_PATH = r"C:\Users\FeiLiu\PycharmProjects\rag_local\models\bge-small-zh-v1.5"
DOC_PATH = r"data/华南师范大学高等学历继续教育学士学位申报系统学生端操作指南.pdf"

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai_like import OpenAILike  # 使用兼容 OpenAI 接口的类
import chromadb

# ==============================================================================
# [1] 嵌入模型配置 — 本地 BGE 中文嵌入模型
#  设置 Embedding 模型（中文场景用 bge-small-zh）
# ==============================================================================
def setup_embed_model():
    """
        配置本地 BGE 中文嵌入模型。
        模型来源: 使用Hugging Face BAAI/bge-small-zh-v1.5
        关键参数说明:
          - model_name: 本地路径，HuggingFaceEmbedding 会自动识别为路径并加载
          - device: "cpu"，因为目标环境无GPU
          - max_length: 128，该模型默认最大文本长度
          - normalize: True，归一化嵌入向量以提高检索精度
          - trust_remote_code: True，本地模型可能包含自定义代码
        中文检索增强策略:
          1. BGE 模型原生支持中文语义编码
          2. 使用中文友好的 SentenceSplitter（按标点分句）
          3. chunk_size 设为 300（中文约150~200字），保证语义完整性
          4. 查询时使用中文查询模板增强检索效果
        """
    # 验证模型路径是否存在
    model_path = Path(EMBEDDING_MODEL_PATH)
    if not model_path.exists():
        logger.error(f"嵌入模型路径不存在: {EMBEDDING_MODEL_PATH}\n"
                     f"请确认模型已下载到该目录，或修改 EMBEDDING_MODEL_PATH 变量")
        sys.exit(1)
    # 检查关键模型文件
    required_files = ["config.json"]
    missing_files = [f for f in required_files if not (model_path / f).exists()]
    if missing_files:
        logger.warning(f"模型目录缺少文件: {missing_files}，尝试继续加载...")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_PATH,  # 直接指向本地目录
        device="cpu",  # CPU 模式
        max_length=128,  # BGE 默认最大长度
        normalize=True,  # 归一化向量
        trust_remote_code=True,  # 允许本地自定义代码
        embed_batch_size=8,  # 小批量避免内存溢出
        show_progress_bar=True,  # 显示嵌入进度条
    )
    # 全局设置嵌入模型
    Settings.embed_model = embed_model
    # 验证嵌入模型
    test_embedding = embed_model.get_text_embedding("测试中文嵌入")
    logger.info(f"嵌入模型配置完成: BGE 中文\n"
                f"  向量维度: {len(test_embedding)}\n"
                f"  模型路径: {EMBEDDING_MODEL_PATH}")
    # 2. 显式设置大语言模型（这里以 DeepSeek 为例，如果是通义千问，换一下 base_url 和 model 名字即可）
    Settings.llm = OpenAILike(
        model="deepseek-flash",
        api_key="你的dsKey",
        api_base="https://api.deepseek.com",
        is_chat_model=True
    )

    return embed_model


# ==============================================================================
# [2] 中文增强的文档分割器
# ==============================================================================
def setup_node_parser():
    """
    配置中文友好的文档分割器。
    中文分割策略:
      - chunk_size: 300 字符（约 150~200 中文字），
        小 chunk 更精确匹配，提升检索召回率
      - chunk_overlap: 50 字符，保证跨 chunk 的语义连贯
      - separator: 保留默认空格（不影响中文切分）
      - paragraph_separator: 段落分隔符
      - secondary_chunking_regex: 加入中文标点的正则，精准断句
    LlamaIndex 的 SentenceSplitter 会按以下优先级分割:
      1. 段落分隔符 (\\n\\n)
      2. 句子级分隔符 (。！？；)
      3. 分词级
      4. 字符级
    """
    node_parser = SentenceSplitter( chunk_size=300, # 中文短 chunk，提升检索精度
        chunk_overlap=50,           # 适当重叠避免语义断裂
        separator=" ",              # 基础分隔符（保留默认）
        paragraph_separator="\n\n", # 段落分隔符
        # 中文标点加入正则，精准断句
        secondary_chunking_regex=r'[^,.;；。？！……，]+[,.;；。？！……，]?|[,.;；。？！……，]', )
    Settings.node_parser = node_parser
    logger.info("文档分割器配置完成: 中文增强 (chunk_size=300)")
    return node_parser

# ==============================================================================
# [3] 加载本地文档（把你准备好的 PDF/Markdown 放进 data/ 目录）
# ==============================================================================
def load_doc():
    doc_path = Path(DOC_PATH)
    if not doc_path.exists():
        logger.error(f"本地文档路径不存在: {DOC_PATH}")
        sys.exit(1)
    # documents = SimpleDirectoryReader("data/").load_data() #简单的读pdf
    # 指定读取 PDF 的解析器
    reader = PyMuPDFReader()
    documents = reader.load_data(file_path=doc_path)
    logger.info(f"本地文档加载完成: 文档路径：{DOC_PATH}")
    return documents

# ==============================================================================
# [4] 创建向量库并建立索引（Chroma 持久化到本地）
# ==============================================================================
def create_vector_store(documents:Sequence[Document]):
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_store = ChromaVectorStore(chroma_collection=db.get_or_create_collection("docs"))
    index = VectorStoreIndex.from_documents(documents, vector_store=chroma_store)
    logger.info("创建向量库并建立索引完成")
    return index

# ==============================================================================
# [5] 问答查询
# ==============================================================================
def rag_local_test(index:VectorStoreIndex,question="文档中关于自考学生的申报流程是什么？"):
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(question)
    # 打印模型参考的上下文内容，看看是不是乱码
    for i, node in enumerate(response.source_nodes):
        print(f"--- 检索到的片段 {i + 1} ---")
        print(node.text)
        print("-" * 30)

    print("最终回答：", response)
    print(response)

def main():
    embed_model = setup_embed_model()
    node_parser = setup_node_parser()
    documents = load_doc()
    index = create_vector_store(documents)
    rag_local_test(index)

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # print_hi('PyCharm')
    main()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
