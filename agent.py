from rag_functions import agent as core_agent


class AgentWrapper:
    def predict(self, request: dict):
        user_input = request["input"][0]["content"]
        result = core_agent(user_input)

        return {"output": result}


AGENT = AgentWrapper()
