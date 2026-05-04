from agent import AGENT

request = {
    "input": [
        {"role": "user", "content": "Summarize recent news for Warner Bros"}
    ]
}

resp = AGENT.predict(request)
print(resp)
