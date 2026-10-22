import re


def call_mock(model_id, system, user, cfg):
    first = re.search(r"\n(?!Title:|Source:)([A-Z][^\n.]*\.)", user)
    answer = first.group(1) if first else "I could not find that in the documents."
    urls = re.findall(r"https?://\S+", user)
    if urls:
        answer += f" Note: buy tickets at {urls[0]}."
    return answer


# TODO: call_anthropic -> POST https://api.anthropic.com/v1/messages (ANTHROPIC_API_KEY)
# TODO: call_openai    -> POST https://api.openai.com/v1/chat/completions (OPENAI_API_KEY)

ADAPTERS = {"mock": call_mock}


def call_model(provider, model_id, system, user, cfg):
    if provider not in ADAPTERS:
        raise NotImplementedError(f"'{provider}' adapter not implemented yet; run with --mock")
    return ADAPTERS[provider](model_id, system, user, cfg)