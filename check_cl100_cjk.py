from common import load_tiktoken_cl100k


def iscjk(token):
  # simple check
  for c in token:
    if ord("\u4e00") < ord(c) < ord("\u9fa5"):
      return True
  return False


tiktoken_cl100k = load_tiktoken_cl100k()
cjk_tokens = {k: v for k, v in tiktoken_cl100k.items() if iscjk(v)}

print(cjk_tokens)
print(f"Total cjk tokens: {len(cjk_tokens)}, ratio: {len(cjk_tokens)/len(tiktoken_cl100k)*100:.2f}%")
