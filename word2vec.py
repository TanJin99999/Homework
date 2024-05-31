import os
import jieba
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import re
import numpy as np
import random
#读取文件夹里的所有文件
def read_files(path):
    files_to_read = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                files_to_read.append(os.path.join(root, file))
    return files_to_read

# 语料预处理，进行断句，去除一些广告和无意义内容
def content_deal(content):  
    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b', '她', '他', '你', '我', '它', '这'] #去掉其中的一些无意义的词语
    for a in ad:
        content = content.replace(a, '')

#计算两个词语义的相似程度
def similarity(model, word1, word2):
    if isinstance(model, Word2Vec):
        vector1 = model.wv[word1]
        vector2 = model.wv[word2]
    else:
        vector1 = model.word_vectors[model.dictionary[word1]]
        vector2 = model.word_vectors[model.dictionary[word2]]
    similarity_12 = cosine_similarity([vector1], [vector2])[0][0]
    return similarity_12


#利用K-Mean聚类判断模型训练的好坏
def plot_clusters(model, num_clusters,):
        # 获取模型中的词汇表
    word_vectors = model.wv
    words = list(word_vectors.key_to_index.keys())
    # 获取词向量
    X = word_vectors[words]
    # 进行K-Means聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(X)
    # 获取聚类结果
    labels = kmeans.labels_
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)
    # 计算轮廓系数
    sil_score = silhouette_score(X, labels)
    print(f'Silhouette Score: {sil_score}')
    # 可视化聚类结果
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Word2Vec K-Means Clustering with t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

#############将文本分断
def get_paragraphs(corpus):
    paragraphs = corpus.split('\n')
    return [p.strip() for p in paragraphs if p.strip()]
#############随机选取段落
def get_random_paragraph(paragraphs):
    return random.choice(paragraphs)

#############将段落的每个词的进行向量平均化获得段落向量值
def get_paragraph_vector(paragraph, model):
    tokens = [word for word in jieba.lcut(paragraph) if word in model.wv] if isinstance(model, Word2Vec) else [word for word in jieba.lcut(paragraph) if word in model.dictionary]
    vectors = [model.wv[token] for token in tokens] if isinstance(model, Word2Vec) else [model.word_vectors[model.dictionary[token]] for token in tokens]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


###############读取并处理语料库
files_to_read=read_files('Resource')
corpus=[]
all_conten=""
for file_name in files_to_read:
    with open(file_name, "r",encoding='gb18030') as file:
        text = file.read()
        #将数据处理可以训练的格式
        sentence_token = [jieba.lcut(sentence) for sentence in text.split('\n') if sentence.strip()]
        corpus.extend(sentence_token)
        all_conten +=text
###############进行模型训练
model = Word2Vec(sentences=corpus, vector_size=200, window=7, min_count=1, workers=10)
print(model.wv.most_similar('杨过',topn=4))
###############计算词向量之间的语意距离
word11=["黄蓉","欧阳锋"]
word1="郭靖"
for word2 in word11:
    similarity_12=similarity(model, word1, word2)
    print(f"'{word1}' 和 '{word2}'的语义距离相似为: {similarity_12}")
word11=["小龙女","尹志平"]
word1="杨过"
for word2 in word11:
    similarity_12=similarity(model, word1, word2)
    print(f"'{word1}' 和 '{word2}'的语义距离相似为: {similarity_12}")
word11=["康熙","吴三桂"]
word1="韦小宝"
for word2 in word11:
    similarity_12=similarity(model, word1, word2)
    print(f"'{word1}' 和 '{word2}'的语义距离相似为: {similarity_12}")

###############词义聚类
plot_clusters(model,num_clusters=3)

##############某些段落直接的语意关联
paragraphs = get_paragraphs(all_conten)
# 从语料库中随机选择两个段落
random_paragraph1 = get_random_paragraph(paragraphs)

random_paragraph2 = get_random_paragraph(paragraphs)

# 计算示例段落的段落向量
paragraph_vector1 = get_paragraph_vector(random_paragraph1, model)
paragraph_vector2 = get_paragraph_vector(random_paragraph2, model)
# 计算示例段落的语义关系
paragraph_similarity = cosine_similarity([paragraph_vector1], [paragraph_vector2])[0][0]
print(random_paragraph1)
print(f"Semantic similarity between paragraphs: {paragraph_similarity}")
print(random_paragraph2)

################计算非语料库的语意关联
random_paragraph3='下面我们通过穿越机的视角，沿长江而下，一起“云游”这跨越1800多公里的长江“绿色生态纽带”。'
random_paragraph4='欢迎总统先生访问中国并出席中阿合作论坛第十届部长级会议开幕式。'

# 计算示例段落的段落向量
paragraph_vector3 = get_paragraph_vector(random_paragraph3, model)
paragraph_vector4 = get_paragraph_vector(random_paragraph4, model)
# 计算示例段落的语义关系
paragraph_similarity = cosine_similarity([paragraph_vector3], [paragraph_vector4])[0][0]
print(random_paragraph3)
print(f"Semantic similarity between paragraphs: {paragraph_similarity}")
print(random_paragraph4)