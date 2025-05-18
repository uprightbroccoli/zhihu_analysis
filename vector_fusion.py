import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import gensim
import gensim.corpora as corpora
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


def load_filtered_words(file_path: str) -> list:
    """
    加载过滤后的分词文本
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 未找到！")
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if line.strip()]


def calculate_coherence_and_perplexity(dictionary, corpus, texts, max_topics=15):
    """
    计算一致性与困惑度，确定最佳主题数
    """
    coherence_values = []
    perplexity_values = []
    for num_topics in range(2, max_topics + 1):
        lda_model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10
        )
        coherence_model_lda = gensim.models.CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_values.append(coherence_model_lda.get_coherence())
        perplexity_values.append(lda_model.log_perplexity(corpus))
    return coherence_values, perplexity_values


def build_lda_vectors(texts: list, best_topic_num: int) -> np.ndarray:
    """
    使用LDA模型生成主题分布向量
    """
    dictionary = corpora.Dictionary([text.split() for text in texts])
    corpus = [dictionary.doc2bow(text.split()) for text in texts]

    # 构建LDA模型
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=best_topic_num,
        random_state=42,
        passes=10
    )
    lda_vectors = np.array([
        np.array([topic_prob[1] for topic_prob in lda_model.get_document_topics(bow, minimum_probability=0)])
        for bow in corpus
    ])

    # 确保每行元素之和为1
    lda_vectors = lda_vectors / lda_vectors.sum(axis=1, keepdims=True)

    return lda_vectors, lda_model


def build_sbert_embeddings(texts: list, model_name: str = "distiluse-base-multilingual-cased-v2") -> np.ndarray:
    """
    使用SBERT生成句子嵌入向量
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def build_autoencoder(input_dim: int, encoding_dim: int = 128):
    """
    构建自编码器模型
    """
    model = Sequential([
        Dense(256, activation="relu", input_dim=input_dim),
        Dense(encoding_dim, activation="relu"),
        Dense(input_dim, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def perform_kmeans_clustering(vectors: np.ndarray, num_clusters: int):
    """
    使用K-means进行聚类并计算轮廓系数
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)
    silhouette_avg = silhouette_score(vectors, labels)
    return labels, silhouette_avg


if __name__ == "__main__":
    filtered_words_file = "C:\\Users\\BeLik\\Desktop\\zhihu_analysis\\output_file/filtered_words.txt"

    # Step 1: Load filtered words
    filtered_texts = load_filtered_words(filtered_words_file)

    # Step 2: LDA topic modeling
    dictionary = corpora.Dictionary([text.split() for text in filtered_texts])
    corpus = [dictionary.doc2bow(text.split()) for text in filtered_texts]
    print("计算一致性与困惑度，确定最佳主题数...")
    coherence_values, perplexity_values = calculate_coherence_and_perplexity(dictionary, corpus, filtered_texts, max_topics=15)
    best_topic_num = np.argmax(coherence_values) + 2
    print(f"最佳主题数: {best_topic_num}")
    lda_vectors, lda_model = build_lda_vectors(filtered_texts, best_topic_num)
    print("LDA主题分布向量构建完成")

    # Step 3: Build SBERT sentence embedding vectors
    sbert_vectors = build_sbert_embeddings(filtered_texts, model_name="distiluse-base-multilingual-cased-v2")
    print("SBERT句子嵌入向量构建完成")

    # Step 4: Feature fusion and dimensionality reduction
    combined_vectors = np.hstack((lda_vectors, sbert_vectors))
    autoencoder = build_autoencoder(input_dim=combined_vectors.shape[1], encoding_dim=128)
    autoencoder.fit(combined_vectors, combined_vectors, epochs=50, batch_size=32, verbose=1)
    fused_vectors = autoencoder.predict(combined_vectors)
    print("特征融合与降维完成")

    # Step 5: Clustering and evaluation
    num_clusters = best_topic_num  # 按照LDA主题数确定K值
    labels, silhouette_avg = perform_kmeans_clustering(fused_vectors, num_clusters)
    print(f"K-means聚类完成, 轮廓系数: {silhouette_avg:.4f}")
    print(f"聚类簇数量: {num_clusters}, 噪声点数: {list(labels).count(-1)}")