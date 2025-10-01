import platform
import sys
import warnings

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core._api import LangChainDeprecationWarning

from log_config import *

# 检查是否是 Windows 系统
if platform.system() == "Windows":
    # 创建伪 fcntl 模块
    class DummyFcntl:
        LOCK_EX = 0x02
        LOCK_SH = 0x01
        LOCK_NB = 0x04
        LOCK_UN = 0x08

        @staticmethod
        def flock(file, operation):
            # Windows 文件锁定占位实现
            pass

        @staticmethod
        def lockf(file, cmd, len=0, start=0, whence=0):
            # Windows 文件锁定占位实现
            pass

    # 注入到 sys.modules
    sys.modules["fcntl"] = DummyFcntl()


# 忽略一些烦人的警告
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


def get_related_data(query, db_path, top_k):
    # 定义一个空列表，用来存储召回的文档块
    related_data_list = []

    # 实例化词嵌入模型
    embedding = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key="sk-39288ed597ab4526ba74c38edb402deb",
    )

    logger.success("词嵌入模型初始化成功！")

    # 载入向量数据库
    db = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
    db.distance_strategy = "COSINE"  # 设置检索时采用余弦相似度
    logger.success(f"向量数据库载入成功！设置使用{db.distance_strategy}进行文档召回！")
    # 到向量数据库中根据用户问题进行检索
    search_res = db.similarity_search(query=query, k=top_k)
    logger.success(f"文档召回成功！{search_res}")
    # 对查询结果进行处理，
    for doc in search_res:
        related_data_list.append(doc.page_content.replace("\n\n", "\n"))

    related_data = "\n".join(related_data_list)

    logger.success("召回文档处理完成！")

    return related_data


if __name__ == "__main__":
    db_path = "knowledge_base/vector_db"
    get_related_data("你好", db_path, 3)
