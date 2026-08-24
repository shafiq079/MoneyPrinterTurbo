import pytest

from app.services.agentrouter import anthropic_messages_url


@pytest.mark.parametrize(
    "base_url",
    ["https://co.agentrouter.org", "https://co.agentrouter.org/v1"],
)
def test_anthropic_base_url_normalization(base_url):
    assert anthropic_messages_url(base_url) == (
        "https://co.agentrouter.org/v1/messages"
    )
