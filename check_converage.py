from common import load_tiktoken_cl100k, load_cjk100k

tiktoken_cl100k = load_tiktoken_cl100k()
cjk100k = load_cjk100k()

tiktoken_set = set(tiktoken_cl100k.values())
for topk in [1000, 10000, len(cjk100k)]:
  cnt = 0
  for i in range(topk):
    if cjk100k[i] in tiktoken_set:
      cnt += 1
  print(f"coverage cjk100k@{topk}", cnt / topk)
