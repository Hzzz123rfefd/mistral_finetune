import nltk
from nltk.translate.bleu_score import sentence_bleu

# 示例文本
references = [
    "The cat is on the mat.",
    "There is a cat on the mat."
]

candidates = [
    "The cat is on mat.",
    "The dog is in the garden."
]

# 分词函数
def tokenize(text):
    return text.lower().split()

# 将参考文本和候选文本分词
references_tokenized = [list(map(tokenize, [ref])) for ref in references]
candidates_tokenized = [tokenize(cand) for cand in candidates]

# 逐对计算 BLEU 分数
bleu_scores = []
for ref, cand in zip(references_tokenized, candidates_tokenized):
    score = sentence_bleu(ref, cand)
    bleu_scores.append(score)

# 计算平均 BLEU 分数
average_bleu = sum(bleu_scores) / len(bleu_scores)

# 输出结果
print("Individual BLEU Scores:", [round(s, 4) for s in bleu_scores])
print(f"Average BLEU Score: {average_bleu:.4f}")
