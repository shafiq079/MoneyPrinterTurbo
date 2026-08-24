"""Small shared helpers for AgentRouter's Anthropic-compatible API."""


def anthropic_messages_url(base_url: str) -> str:
    """Normalize documented root and versioned base URLs to Messages API."""
    root = base_url.strip().rstrip("/")
    if not root.endswith("/v1"):
        root += "/v1"
    return f"{root}/messages"
