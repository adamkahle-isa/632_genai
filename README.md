# 632_genai
Using GenAI in business

## Databricks Agent Deployment (Module08 Pattern)

### 1) Prepare your core logic module
Your existing functions (`get_articles`, `count_articles`, `get_company_info`, `summarize_company`, `agent`) must be importable from a Python file (for example `rag_functions.py`).

`agent.py` imports:

```python
from rag_functions import agent
```

If your module name differs, update that single import line.

### 2) Test local adapter object

```python
from agent import AGENT

request = {
    "input": [
        {"role": "user", "content": "Summarize recent news for Warner Bros"}
    ]
}

print(AGENT.predict(request))
```

### 3) Log + register to Unity Catalog

Run:

```bash
python log_and_register_agent.py
```

This script:
- sets `mlflow.set_registry_uri("databricks-uc")`
- sets model name to `isa632_7474656346303369.fladunhh.genai_project`
- calls `mlflow.models.set_model(AGENT)`
- logs with `mlflow.pyfunc.log_model(..., python_model="agent.py")`
- registers with `mlflow.register_model(...)`

### 4) Deploy as Model Serving endpoint
1. In Databricks, go to **Serving** → **Create serving endpoint**.
2. Select the registered Unity Catalog model `isa632_7474656346303369.fladunhh.genai_project` and desired version.
3. Choose compute size and click **Create**.
4. Wait until endpoint status is **Ready**.

### 5) Enable in Databricks Agents UI
1. Open the Databricks **Agents** UI.
2. Choose **Custom agent** (or add an existing endpoint, depending on workspace UI version).
3. Select the serving endpoint you created from the registered model.
4. Save/publish the agent configuration.

### 6) Test in Agents UI
- Open the agent chat panel.
- Send: `Summarize recent news for Warner Bros`.
- Confirm the response is returned from your deployed endpoint.
