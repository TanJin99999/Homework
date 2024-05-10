import  os
import numpy as np
from sklearn.model_selection import train_test_split
import os
import jieba
import random
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import precision_score, f1_score
from collections import Counter 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
############################################################################################################################################
#读取文件夹里的所有文件
def read_files(path):
    files_to_read = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                files_to_read.append(os.path.join(root, file))
    return files_to_read

#从语料库中抽取数据集
def extract_dataset(corpus, labels, num_paragraphs):
    dataset = []
    dataset_labels = []
    for label in set(labels):
        # 确定某文档在语料库中占的比例，进行均匀配额
        lst = labels
        count = lst.count(label)
        #确定某文档要抽取的段落数
        num_paragraphs_per_label = int(num_paragraphs * count / len(labels))
        label_paragraphs = [paragraph for paragraph, paragraph_label in zip(corpus, labels) if paragraph_label == label]
        sampled_paragraphs = np.random.choice(label_paragraphs, num_paragraphs_per_label, replace=False)
        dataset.extend(sampled_paragraphs)
        dataset_labels.extend([label] * num_paragraphs_per_label)
    return dataset, dataset_labels

#对文章内容进行处理，将文章里停用词去掉
def content_deal(paragraph,stopwords):
    return [word for word in paragraph if word not in stopwords]

#将小说文件分段
def extract_paragraphs(novels, paragraph_len,max_length):
    paragraphs = []
    labels = []
    for label, text in novels.items():
        words = []
        start_index = 0
        if (max_length > 1):
                words = list(jieba.cut(text))
        else:
            while start_index < len(text):
                words.append(text[start_index])
                start_index += 1
        words=content_deal(words,stopwords)
        paragraphs += [' '.join(words[i:i+paragraph_len]) for i in range(0, len(words), paragraph_len)]
        labels += [label] * ((len(words) // paragraph_len)+1)
    return paragraphs, labels

#对训练集合进行LDA建模训练
def preprocess_with_lda(X_train, X_test, K, T,max_length):
    if max_length == 2:
        analyzer = 'word'
    elif max_length == 1:
        analyzer = 'char'
    lda_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(max_features=K, analyzer=analyzer)),
        ('lda', LatentDirichletAllocation(n_components=T, random_state=42))
    ])
    X_train_lda = lda_pipeline.fit_transform(X_train)
    X_test_lda = lda_pipeline.transform(X_test)

    return X_train_lda, X_test_lda


#对训练结果分类并进行评估
def classify_and_evaluate(X_train_lda, y_train, X_test_lda, y_test,num_cross,classifier):
    #选择分类器
    if classifier == 'LogisticRegression':
         clf = LogisticRegression(random_state=514)
    elif classifier == 'MultinomialNB':
        clf = MultinomialNB()
    elif classifier == 'SVC':
        clf = SVC(random_state=514)
    elif classifier == 'RandomForestClassifier':
        clf = RandomForestClassifier(random_state=100)
    else:
        raise ValueError("Unsupported classifier type")
    #交叉验证
    clf.fit(X_train_lda, y_train)
    
    accuracy = np.mean(cross_val_score(clf, X_train_lda, y_train, cv=num_cross))
    accuracy_test = np.mean(cross_val_score(clf, X_test_lda, y_test, cv=num_cross))
    
    return accuracy,accuracy_test





############################################################################################################################################################
#读取停用词
files_to_read=read_files('tinyonci')
for file_name in files_to_read:
    with open(file_name, "r",encoding='utf-8') as file:
     stopwords_text=file.read()
     stopwords = stopwords_text.split('\n')
     stopwords+=['\n','\u3000',' ']
#枚举文件夹里的所有文件
files_to_read1=read_files('Resource')
#构建语料库，corpus是段落列表，labels是段落所属小说标签列表
novels={}
for file_name1 in files_to_read1:
    with open(file_name1, "r",encoding='ANSI') as file:
        text = file.read()
        #只保留file_name1的中文字符部分
        file_name1 = file_name1.split('.')[0]
        file_name1 = file_name1.split('\\')[-1]
        novels[file_name1] = text

#设置参数
K1=[1000]#token数
max_length1=[2]#以词为单位2，以字为单位1
num_topics1= [40]#主题数
classifie1=['RandomForestClassifier']#['SVC','RandomForestClassifier','MultinomialNB','LogisticRegression']选择四种分类器


#对文本进行分段，以K个token为一个段落
#K1=[20,100,500,1000,2000,3000]
for K in K1:
    K=K
    #以词为单位2，以字为单位1
    
    for max_length in max_length1:
        max_length=max_length
        paragraphs, labels = extract_paragraphs(novels, K, max_length)
    #均匀抽取1000个段落作为数据集
        num_paragraphs=1000
        dataset, dataset_labels = extract_dataset(paragraphs, labels, num_paragraphs)

    #将数据集划分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(dataset, dataset_labels, test_size=0.1, random_state=50)

    #对训练集进行LDA主题建模
    
        for num in num_topics1:
            num_topics =num
            X_train_lda, X_test_lda = preprocess_with_lda(X_train, X_test, K,num_topics,max_length)
            
            #对训练结果进行分类并交叉检验评估
            for classifier in classifie1:
                classifier=classifier
                accuracies,accuracies_test=classify_and_evaluate(X_train_lda, y_train, X_test_lda, y_test,num_cross=10,classifier=classifier)
                #打印并换行
                print(f"获取段落数: {num_paragraphs}")
                print(f"Token: {K}")
                print(f"主题数量: {num_topics}")
                print(f"字或词: {max_length}")
                print(f"分类器: {classifier}")
                print(f"训练精度: {accuracies}")
                print(f"测量精度: {accuracies_test}")