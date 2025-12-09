import tiktoken
from app.core.config import settings

def count_tokens(text: str) -> int:
    """
    估算 Token 数量。
    对于中文，tiktoken 估算可能不准，但作为截断依据足够。
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        # 简单回退：中文 1 char ~ 1 token
        return len(text)

def truncate_text(text: str, max_tokens: int) -> str:
    """
    截断文本以适应 Token 限制
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        return encoding.decode(tokens[:max_tokens])
    return text
