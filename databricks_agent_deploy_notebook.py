# Databricks notebook source
# CELL 1 — Setup Path
import sys

sys.path.append("/Workspace/Users/fladunhh@miamioh.edu/")
print("Path configured")

# COMMAND ----------
# CELL 2 — Validate Tables
spark.sql("SELECT COUNT(*) FROM news_articles").show()
spark.sql("SELECT COUNT(*) FROM company_data").show()

# COMMAND ----------
# CELL 3 — Run Test Script
print("Running test_agent...")
import importlib
import test_agent

importlib.reload(test_agent)
print("test_agent complete")

# COMMAND ----------
# CELL 4 — Log + Register Model
print("Running log_and_register_agent...")
import log_and_register_agent

# Ensure script main() executes inside notebook import flow
if hasattr(log_and_register_agent, "main"):
    log_and_register_agent.main()

print("Model logged and registered")

# COMMAND ----------
# CELL 5 — Final Agent Test (Optional)
from agent import AGENT

request = {
    "input": [
        {"role": "user", "content": "Summarize recent news for Warner Bros"}
    ]
}

response = AGENT.predict(request)
print(response)
