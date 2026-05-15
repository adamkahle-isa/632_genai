# Databricks notebook source
# MAGIC %md
# MAGIC ## Lab - Build and Register a Finance RAG Model
# MAGIC
# MAGIC In this lab, we construct a RAG pipeline for the Finance Bot and register it in the Unity Catalog model registry.
# MAGIC The RAG pipeline functions as a **due diligence research assistant** for public companies, using news articles
# MAGIC as contextual information retrieved via Vector Search.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC 1. Set up Vector Search endpoint and index for financial news retrieval.
# MAGIC 2. Build a RAG pipeline using LangGraph and the Agent Framework.
# MAGIC 3. Register the RAG model in Unity Catalog for versioning and deployment.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# MAGIC %pip install -U \
# MAGIC   "databricks-langchain" \
# MAGIC   "databricks-agents" \
# MAGIC   "langchain" \
# MAGIC   "mlflow>=3.1" \
# MAGIC   "langgraph>=1.1.4" \
# MAGIC   "langgraph-prebuilt>=1.0.10"
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# pip install --upgrade langgraph --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-02

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Set the Vector Search endpoint, source table, and index variables.

# COMMAND ----------

# assign vs search endpoint by username
vs_endpoint_prefix = "ai_agent_vs_endpoint_"
vs_endpoint_name = vs_endpoint_prefix+'kahlear' # replace with your MUID
vs_source_table_name = "isa632_7474656346303369.kahlear.pdf_docs" # replace with your MUID
VS_INDEX_NAME = "isa632_7474656346303369.kahlear.pdf" # replace with your MUID

print(f"=== Finance Bot Variables ===\n")
print(f"VS Endpoint Name   : {vs_endpoint_name}")
print(f"VS Source Table    : {vs_source_table_name}")
print(f"VS Index Name      : {VS_INDEX_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Dataset
# MAGIC
# MAGIC Display the news source table used for Vector Search retrieval.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Vector Index
# MAGIC
# MAGIC Before building the RAG pipeline, verify the index is ready by searching for similar news articles.

# COMMAND ----------

# DBTITLE 1,Cell 11
from databricks.vector_search.client import VectorSearchClient
from pprint import pprint

vsc = VectorSearchClient(disable_notice=True)

question = "What is happening in Q4"

try:
    index = vsc.get_index(vs_endpoint_name, VS_INDEX_NAME)
    results = index.similarity_search(
        query_text=question,
        columns=["text"],
        num_results=4
    )
    docs = results.get("result", {}).get("data_array", [])
    pprint(docs)
except Exception as e:
    print(f"Error occurred while loading the index: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable MLflow Tracing

# COMMAND ----------

import mlflow
mlflow.langchain.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the RAG Model
# MAGIC
# MAGIC Load the agent from `agent.py` and test it with a sample query.
# MAGIC The agent uses LangGraph's `create_react_agent` with the news retriever tool.

# COMMAND ----------

import os
os.environ["VS_INDEX_NAME"] = VS_INDEX_NAME

# COMMAND ----------

# Load the agent code from agent.py
from agent import AGENT

user_input = "What is happening in Q4"
request = {
    "input": [
        {"role": "user", "content": user_input}
    ]
}
resp = AGENT.predict(request)
print(resp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the RAG Model to Unity Catalog
# MAGIC
# MAGIC Register the Finance Bot model in Unity Catalog for versioning and deployment.

# COMMAND ----------

import mlflow
from mlflow.models.resources import DatabricksVectorSearchIndex
from pkg_resources import get_distribution

# Set Model Registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

model_name = "isa632_7474656346303369.kahlear.finbot" # replace with your MUID

# Set the agent
mlflow.models.set_model(AGENT)

# Register the assembled RAG model in Model Registry with Unity Catalog
with mlflow.start_run(run_name="finbot_rag_register") as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        pip_requirements=[
            f"langchain=={get_distribution('langchain').version}",
            f"databricks-vectorsearch=={get_distribution('databricks-vectorsearch').version}",
            f"databricks_langchain=={get_distribution('databricks_langchain').version}",
            f"mlflow=={get_distribution('mlflow').version}"
        ],
        resources=[
            DatabricksVectorSearchIndex(index_name=VS_INDEX_NAME)
        ]
    )
model_uri = f"runs:/{run.info.run_id}/{model_info.name}"
model_version = mlflow.register_model(model_uri, model_name)
print(f"Model registered with name: {model_name} and version: {model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, we:
# MAGIC
# MAGIC 1. Configured Vector Search for financial news retrieval.
# MAGIC 2. Built a RAG pipeline using LangGraph with a news retriever tool and finance-specific system prompt.
# MAGIC 3. Used an agent-as-code approach with `agent.py` for modular development.
# MAGIC 4. Registered the Finance Bot in Unity Catalog for versioning and deployment.