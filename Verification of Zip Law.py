import re
import matplotlib.pyplot as plt
from collections import Counter
import jieba
import os
# 输入中文语料
folder_path = 'Resource' #输入你文件所在的文件夹
corpus = ""
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='ANSI') as file:
           corpus += file.read()

#with open('D:\(1)\笑傲江湖.txt ', 'r', encoding='ANSI') as f:
  #  text = f.read()

#提取语料中的中文
text_processed = re.sub(r'[^\u4e00-\u9fa5]', '', corpus)
#结巴分词
words =jieba.cut(text_processed)
# 计算汉字频率
word_freq = Counter(words)
# 根据频率排序
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
# 绘制排名和频率的关系图
ranks = range(1, len(sorted_word_freq)+1)
freqs = [freq for _, freq in sorted_word_freq]
plt.loglog(ranks, freqs)
plt.xlabel('rankings of words', fontsize=14, fontweight='bold', fontproperties='SimHei')
plt.ylabel('Frequency of words', fontsize=14, fontweight='bold', fontproperties='SimHei')
plt.title('The relationship between frequency and word ranking')
plt.grid(True)
plt.show()
