import sys
import warnings
from os import path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core._api import LangChainDeprecationWarning
from loguru import logger

from log_config import *

# 获取当前脚本的目录 (know 文件夹)
current_dir = path.dirname(path.abspath(__file__))

# 获取父目录 (know 文件夹的上一级目录)
parent_dir = path.dirname(current_dir)

# 将父目录添加到 sys.path
sys.path.insert(0, parent_dir)  # 或者使用 sys.path.append(parent_dir)


# 忽略一些烦人的警告
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


def create_vector_db(source_data, db_path):
    # 1. 读取本地数据源文件
    logger.info("开始读取数据源文件...")
    loader = UnstructuredFileLoader(source_data)
    doc = loader.load()
    logger.success("数据源文件读取成功！")
    # 2. 对读取的文档进行切分
    logger.info("开始切分文档...")
    doc_spliter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=300)
    split_doc = doc_spliter.split_documents(doc)

    # 切分后的文档是一个 List，这里我们可以查看第一个元素，该元素是一个 Document 对象，我们使用 page_content 属性读取第一个文本块的内容进行查看
    logger.success("文档切分成功！")

    # 3. 将切分后的文档进行向量化
    logger.info("开始初始化词嵌入模型...")
    embedding = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key="sk-39288ed597ab4526ba74c38edb402deb",
    )
    logger.success("词嵌入模型初始化成功！开始构建向量数据库...")
    db = FAISS.from_documents(
        split_doc,
        embedding,
        distance_strategy="COSINE",  # 指定使用余弦相似度，如果不设置，则默认使用欧氏距离
    )

    db.save_local(db_path)
    logger.success("向量数据库构建成功！")


if __name__ == "__main__":
    source_data = "RAG 项目知识库文档.md"
    db_path = "vector_db"
    create_vector_db(source_data, db_path)
