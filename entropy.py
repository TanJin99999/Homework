import os
import jieba
import math

files_to_read = []


def enum_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                files_to_read.append(os.path.join(root, file))


def get_unigram_tf(split_words):
    unigram_tf = {}
    for word in split_words:
        unigram_tf[word] = unigram_tf.get(word, 0) + 1
    return unigram_tf


def get_bigram_tf(split_words):
    bigram_tf = {}
    for i in range(len(split_words) - 1):
        bigram_tf[(split_words[i], split_words[i + 1])] = bigram_tf.get((split_words[i], split_words[i + 1]), 0) + 1
    return bigram_tf


def get_trigram_tf(split_words):
    trigram_tf = {}
    for i in range(len(split_words) - 2):
        trigram_tf[(split_words[i], split_words[i + 1], split_words[i + 2])] = trigram_tf.get(
            (split_words[i], split_words[i + 1], split_words[i + 2]), 0) + 1
    return trigram_tf


def calc_entropy_unigram(split_words):
    word_ft = get_unigram_tf(split_words)
    word_len = sum([intem[1] for intem in word_ft.items()])
    entropy = sum([-(intem[1] / word_len) * math.log2(intem[1] / word_len) for intem in word_ft.items()])
    return entropy


def calc_entropy_bigram(split_words):
    word_tf = get_bigram_tf(split_words)
    last_word_tf = get_unigram_tf(split_words)
    bigram_len = sum([intem[1] for intem in word_tf.items()])
    entropy = []
    for bigram in word_tf.items():
        p_xy = bigram[1] / bigram_len
        p_x_y = bigram[1] / last_word_tf[bigram[0][0]]
        entropy.append(-p_xy * math.log2(p_x_y))
    entropy = sum(entropy)
    return entropy


def calc_entropy_trigram(split_words):
    word_tf = get_trigram_tf(split_words)
    last_word_tf = get_bigram_tf(split_words)
    trigram_len = sum([intem[1] for intem in word_tf.items()])
    entropy = []
    for trigram in word_tf.items():
        p_xy = trigram[1] / trigram_len
        p_x_y = trigram[1] / last_word_tf[(trigram[0][0], trigram[0][1])]
        entropy.append(-p_xy * math.log2(p_x_y))
    entropy = sum(entropy)
    return entropy


# 枚举Resource文件夹下的所有txt文件
enum_files('Resource')
all_content = ""
for file in files_to_read:
    # 读取文件内容
    with open(file, 'r', encoding='ANSI') as f:
        content = f.read()
        # 删除所有非中文字符，以及和小说内容无关的片段
        content = ''.join([c for c in content if '\u4e00' <= c <= '\u9fa5'])
        content = content.replace('本书来自免费小说下载站更多更新免费电子书请关注', '')
        all_content += content

        split_words1 = [word for word in jieba.cut(content)]

        split_words = [word for word in content]

        word_len = content.__len__()

        # 计算unigram的tf
        unigram_tf1 = get_unigram_tf(split_words1)

        # 计算unigram的熵
        entropy_unigram1 = calc_entropy_unigram(split_words1)
        entropy_unigram = calc_entropy_unigram(split_words)
        # 计算bigram的tf
        # bigram_tf = get_bigram_tf(split_words)
        # 计算bigram的熵
        entropy_bigram1 = calc_entropy_bigram(split_words1)
        entropy_bigram = calc_entropy_bigram(split_words)
        # 计算trigram的tf
        # trigram_tf = get_trigram_tf(split_words)
        # 计算trigram的熵
        entropy_trigram1 = calc_entropy_trigram(split_words1)
        entropy_trigram = calc_entropy_trigram(split_words)

        print('文件名: %s' % file)
        print('unigram熵（字）: %f' % entropy_unigram)
        print('bigram熵（字）: %f' % entropy_bigram)
        print('trigram熵（字）: %f' % entropy_trigram)
        print('unigram熵（词-）: %f' % entropy_unigram1)
        print('bigram熵（词）: %f' % entropy_bigram1)
        print('trigram熵（词）: %f' % entropy_trigram1)
        # print('-----------------------------------')

split_words1 = [word for word in jieba.cut(all_content)]
split_words = [word for word in all_content]
all_content_entropy_unigram1 = calc_entropy_unigram(split_words1)
all_content_entropy_unigram = calc_entropy_unigram(split_words)
all_content_entropy_bigram1 = calc_entropy_bigram(split_words1)
all_content_entropy_bigram = calc_entropy_bigram(split_words)
all_content_entropy_trigram1 = calc_entropy_trigram(split_words1)
all_content_entropy_trigram = calc_entropy_trigram(split_words)

print('所有文件：' )
print('unigram熵（字）: %f' % all_content_entropy_unigram)
print('bigram熵（字）: %f' % all_content_entropy_bigram)
print('trigram熵（字）: %f' % all_content_entropy_trigram)
print('unigram熵（词-）: %f' % all_content_entropy_unigram1)
print('bigram熵（词）: %f' % all_content_entropy_bigram1)
print('trigram熵（词）: %f' % all_content_entropy_trigram1)

print('所有文件处理完毕')























