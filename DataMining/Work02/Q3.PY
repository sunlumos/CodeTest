import os, sys

os.chdir(sys.path[0])

# 定义最小支持度
min_support = 0.1

# 读取vocb.txt文件，将词汇映射到word
vocb = {}
with open('./dataset/vocab.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        term, word = line.strip().split('\t')
        vocb[term] = word

# 定义函数，用于计算支持度
def get_support(transactions, itemset):
    count = 0
    for transaction in transactions:
        if all(term in transaction for term in itemset):
            count += 1
    return count / len(transactions)

# 定义函数，用于生成候选项集
def generate_candidates(itemsets, k):
    candidates = set()
    for i in range(len(itemsets)):
        for j in range(i + 1, len(itemsets)):
            candidate = itemsets[i].copy()
            candidate.update(itemsets[j])
            if len(candidate) == k:
                candidates.add(frozenset(candidate))
    return candidates


# 读取每个主题的文件，生成事务列表
transactions = []
for i in range(5):
    with open(f'./dataset/topic-{i}.txt', 'r') as f:
        lines = f.readlines()
        transaction = set()
        for line in lines:
            terms = line.strip().split(' ')
            transaction |= set(terms)
        transactions.append(transaction)

# 初始化频繁1项集
frequent_itemsets = [{frozenset([term]): 0} for term in vocb.keys()]

# 迭代生成频繁k项集
k = 1
while True:
    candidates = generate_candidates(frequent_itemsets, k)
    frequent_itemsets_k = {}
    for candidate in candidates:
        support = get_support(transactions, candidate)
        if support >= min_support:
            frequent_itemsets_k[candidate] = support
    if not frequent_itemsets_k:
        break
    frequent_itemsets.append(frequent_itemsets_k)
    k += 1

# 按支持度从高到低排序并输出到文件
output_dir = 'pattern'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
for i, frequent_itemsets_k in enumerate(frequent_itemsets):
    with open(os.path.join(output_dir, f'pattern-{i}.txt'), 'w') as f:
        for itemset, support in sorted(frequent_itemsets_k.items(), key=lambda x: x[1], reverse=True):
            pattern = ' '.join([vocb[term] for term in itemset])
            f.write(f'{support:.3f} [{pattern}]\n')
