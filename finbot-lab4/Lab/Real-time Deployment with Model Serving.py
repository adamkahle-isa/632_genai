# Databricks notebook source
# MAGIC %md
# MAGIC # Lab - Real-time Deployment with Model Serving
# MAGIC
# MAGIC In this lab, we serve the Finance Bot model using Mosaic Model Serving and the Agent Framework.
# MAGIC When models are served with the Agent Framework, a Review App is automatically deployed alongside
# MAGIC the model, allowing you to interact with the Finance Bot and gather human feedback on its responses.
# MAGIC
# MAGIC ### Learning Objectives
# MAGIC
# MAGIC 1. Deploy the Finance Bot using the Agent Framework.
# MAGIC 2. Interact with the deployed model using the Review App.
# MAGIC 3. Collect and review human feedback on model responses.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# MAGIC %pip install databricks-langchain databricks-agents langchain langgraph mlflow>=3.1 --upgrade --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

pip install --upgrade langgraph --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# assign vs search endpoint by username
vs_endpoint_prefix = "ai_agent_vs_endpoint_"
vs_endpoint_name = vs_endpoint_prefix+'kahlear' # replace with your MUID
vs_source_table_name = "isa632_7474656346303369.kahlear.pdf_docs" # replace with your MUID
VS_INDEX_NAME = "isa632_7474656346303369.kahlear.pdf" # replace with your MUID

model_name = "isa632_7474656346303369.kahlear.finbot" # replace with your MUID

print(f"=== Finance Bot Deployment Variables ===\n")
print(f"Model Name         : {model_name}")
print(f"VS Index Name      : {VS_INDEX_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Deploy the Model with Agent Framework
# MAGIC
# MAGIC This deploys the registered Finance Bot model as a serving endpoint with a Review App
# MAGIC for collecting human feedback.

# COMMAND ----------

# DBTITLE 1,Cell 9
import sys
import time
import mlflow

# Clear stale databricks module cache to allow namespace package resolution
for _key in list(sys.modules.keys()):
    if _key.startswith("databricks"):
        del sys.modules[_key]

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks import agents

# Deploy the model with the agent framework, passing VS_INDEX_NAME as an environment variable
try:
    deployment_info = agents.deploy(
        model_name, 
        model_version=1,
        scale_to_zero=True,
        environment_vars={"VS_INDEX_NAME": VS_INDEX_NAME}
    )
except ValueError as e:
    if "already serves model" not in str(e):
        raise
    deployment_info = agents.get_deployments(model_name=model_name, model_version=1)
    if isinstance(deployment_info, list):
        deployment_info = deployment_info[0]

# Wait for the Review App and deployed model to be ready
w = WorkspaceClient()
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")

while ((w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY) or (w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS)):
    print(".", end="")
    time.sleep(30)

print("\nThe endpoint is ready!", end="")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Access the Review App
# MAGIC
# MAGIC The Review App allows stakeholders to interact with the Finance Bot, ask questions about
# MAGIC public companies, and provide feedback on the quality of responses.

# COMMAND ----------

print(f"Endpoint URL    : {deployment_info.endpoint_url}")
print(f"Review App URL  : {deployment_info.review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Collect Human Feedback via Review App
# MAGIC
# MAGIC The Databricks Review App stages the Finance Bot in an environment where expert stakeholders 
# MAGIC can engage with it. This enables:
# MAGIC
# MAGIC * Conversations about public companies and market events
# MAGIC * Feedback on response accuracy, tone, and completeness
# MAGIC * Identification of hallucinations or incorrect information
# MAGIC * Assessment of whether the bot appropriately refuses investment recommendations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this lab, we:
# MAGIC
# MAGIC 1. Deployed the Finance Bot model using the Agent Framework with `agents.deploy()`.
# MAGIC 2. Configured the VS_INDEX_NAME environment variable for the serving endpoint.
# MAGIC 3. Accessed the Review App URL for human-in-the-loop evaluation.
# MAGIC 4. Learned how to collect feedback on the Finance Bot's responses for continuous improvement.