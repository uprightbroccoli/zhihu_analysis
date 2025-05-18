import pandas as pd
import hashlib
import re
import jieba
from typing import List, Optional, Tuple


def load_csv(file_path: str) -> pd.DataFrame:
    """
    读取 CSV 文件，并返回 DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        print("初始数据基本信息：")
        df.info()
        unique_question_count = df["问题标题"].nunique()
        print(f"问题标题的数量：{unique_question_count}")
        return df
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return pd.DataFrame()
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return pd.DataFrame()


def read_text_file(file_path: str) -> Optional[List[str]]:
    """
    读取文本文件内容，每一行构成一个列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [line.strip() for line in lines if line.strip()]
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None


def write_text_file(file_path: str, content: List[str]) -> None:
    """
    将内容列表写入文本文件，每个元素一行
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for line in content:
                file.write(line + "\n")
        print(f"数据已成功保存为 '{file_path}'")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")


def calculate_hash(text: str) -> str:
    """
    计算文本的 SHA-256 哈希值
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text):
    """
    清洗文本内容
    """
    if pd.isnull(text):
        return ""

    text = re.sub(r'<.*?>', '', text)  # HTML标签
    text = re.sub(r'&[a-z]+;', '', text)  # HTML转义字符
    text = re.sub(r'http\S+|www\S+', '', text)  # URL链接
    text = re.sub(r'pic\.\S+|img\.\S+', '', text)  # 图片链接
    text = re.sub(r'↓', '', text)
    text = re.sub(r'\d+', '', text)  # 移除数字
    text = re.sub(r'\s+', ' ', text)  # 多余空白符
    return text.strip()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据预处理
    """
    # 处理赞同数和评论数
    df["赞同数"] = df["赞同数"].astype(str).str.replace(
        r"\s|​|,|个|👍|赞|\+", "", regex=True
    )
    df["评论数"] = df["评论数"].astype(str).str.replace(
        r"\s|​|,|条|评论", "", regex=True
    )

    # 转换为数字类型，无法转换的设为 NaN
    df["赞同数"] = pd.to_numeric(df["赞同数"], errors="coerce")
    df["评论数"] = pd.to_numeric(df["评论数"], errors="coerce")

    # 标准化“回答时间”字段
    df["回答时间"] = df["回答时间"].astype(str).str.extract(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})")
    df["回答时间"] = pd.to_datetime(df["回答时间"], errors="coerce")

    # 删除完全缺失“回答内容”的记录
    df = df.dropna(subset=["回答内容"])

    # 填充其他缺失值
    df.loc[:, "问题内容"] = df["问题内容"].fillna("")
    df.loc[:, "答主昵称"] = df["答主昵称"].fillna("匿***")
    df.loc[:, "回答时间"] = df["回答时间"].fillna("2025-05-08 16:23:00")
    df.loc[:, "赞同数"] = df["赞同数"].fillna("0")
    df.loc[:, "评论数"] = df["评论数"].fillna("0")

    return df


def filter_and_group(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    数据过滤与分组
    """
    # 应用文本清洗到“回答内容”
    df["回答内容"] = df["回答内容"].apply(clean_text)

    # 添加哈希列，用于后续数据去重
    df["content_hash"] = df["回答内容"].apply(calculate_hash)

    # 检查并转换赞同数为数值类型
    df["赞同数"] = pd.to_numeric(df["赞同数"], errors="coerce").fillna(0)

    # 按赞同数排序并去重
    df = (
        df.sort_values("赞同数", ascending=False)
        .drop_duplicates(subset="content_hash")
        .drop(columns=["content_hash"])
        .reset_index(drop=True)
    )

    print("数据处理完成！")

    # 剔除回答内容少于15字的记录
    df = df[df["回答内容"].str.len() >= 15]
    df = df.reset_index(drop=True)
    print(f"剔除过短评论后剩余 {len(df)} 条记录")

    # 按照“问题标题”对回答内容合并
    df_grouped = (
        df.groupby("问题标题")["回答内容"].apply(lambda x: " ".join(x)).reset_index()
    )

    # 对每个问题统计回答数
    question_counts = df["问题标题"].value_counts().reset_index()
    question_counts.columns = ["问题标题", "回答数"]
    question_counts_sorted = question_counts.sort_values(by="回答数", ascending=False)

    return df, df_grouped, question_counts_sorted


def segment_text_with_dict(raw_words_file: str, dict_file: str, output_file: str) -> None:
    """
    使用自定义词典对文本进行分词，并将结果保存到文件
    """
    # 加载自定义词典
    jieba.load_userdict(dict_file)

    # 读取原始文本
    raw_words = read_text_file(raw_words_file)
    if not raw_words:
        print(f"错误: 无法读取文件 '{raw_words_file}' 或文件内容为空。")
        return

    # 对文本进行分词
    segmented_words = [" ".join(jieba.cut(line)) for line in raw_words]

    # 保存分词结果
    write_text_file(output_file, segmented_words)


def remove_stopwords(segmented_file: str, stopwords_file: str, output_file: str) -> None:
    """
    使用停用词表对分词后的文本进行去除停用词处理
    """
    # 读取停用词表
    stopwords = set(read_text_file(stopwords_file) or [])
    if not stopwords:
        print(f"错误: 无法加载停用词表 '{stopwords_file}' 或文件内容为空。")
        return

    # 读取分词结果
    segmented_words = read_text_file(segmented_file)
    if not segmented_words:
        print(f"错误: 无法读取文件 '{segmented_file}' 或文件内容为空。")
        return

    # 去除停用词
    filtered_words = [
        " ".join(word for word in line.split() if word not in stopwords)
        for line in segmented_words
    ]

    # 保存处理后的结果
    write_text_file(output_file, filtered_words)


if __name__ == "__main__":
    input_file = "feet_file\\raw_data.csv"
    cleaned_words_file = "feet_file\\raw_words.txt"
    dict_file = "tool_file\\diydict.txt"
    segmented_words_file = "output_file\\segmented_words.txt"
    stopwords_file = "tool_file\\stopwords.txt"
    filtered_words_file = "output_file\\filtered_words.txt"

    # Load the data
    df = load_csv(input_file)

    # Preprocess the data
    df = preprocess_data(df)

    # Perform filtering and grouping
    cleaned_df, grouped_df, question_counts = filter_and_group(df)

    # Save the cleaned "回答内容" to a text file
    write_text_file(cleaned_words_file, cleaned_df["回答内容"].tolist())

    # Perform segmentation using the custom dictionary
    segment_text_with_dict(cleaned_words_file, dict_file, segmented_words_file)

    # Remove stopwords from the segmented text
    remove_stopwords(segmented_words_file, stopwords_file, filtered_words_file)

    # Print the top 10 question titles and their respective answer counts
    print("前10个问题标题及其回答数:")
    print(question_counts.head(10))

