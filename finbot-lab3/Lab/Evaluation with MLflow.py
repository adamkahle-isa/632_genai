# Databricks notebook source
# MAGIC %md
# MAGIC # Lab - Evaluation with MLflow
# MAGIC
# MAGIC After building the Finance Bot model, we evaluate its performance using the Agent Framework.
# MAGIC This lab demonstrates how to calculate built-in evaluation metrics and define custom metrics
# MAGIC tailored to financial due diligence responses.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC * Load the Finance Bot model for evaluation.
# MAGIC * Define custom evaluation metrics for rigor and accuracy.
# MAGIC * Run an evaluation test and view results using MLflow UI.

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Agent
# MAGIC
# MAGIC We build the agent directly (same as in Lab 2) so we can use it for evaluation.

# COMMAND ----------

import os
import mlflow
from uuid import uuid4
from typing import Any, List, Dict

from mlflow.pyfunc import ResponsesAgent
from mlflow.entities import SpanType
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool


def build_agent():
    vs_index_name = os.environ.get("VS_INDEX_NAME")
    if not vs_index_name:
        raise RuntimeError("VS_INDEX_NAME is not set in environment")

    llm = ChatDatabricks(
        endpoint=os.environ.get("LLM_ENDPOINT_NAME", "databricks-llama-4-maverick"),
        max_tokens=500,
        temperature=0.5,
    )

    retriever_tool = VectorSearchRetrieverTool(
        name="news_search_tool",
        index_name=vs_index_name,
        description=(
            "Use to find relevant news across multiple sources to help inform investments "
            "Input should be a natural-language question, company names, or keywords."
        ),
    )

    system_prompt = """You are an FDIC banking-industry analysis assistant grounded in the provided Quarterly Banking Profile document.

PURPOSE
- Answer questions about trends, risks, opportunities, financial performance, asset quality, deposits, lending, capital, and banking-sector conditions described in the report.
- Provide concise analytical summaries based on the document.
- Support responses with metrics, trends, and observations from the report whenever possible.

SYSTEM BEHAVIOR
- Treat each query independently.
- Use retrieved document context as the primary source of truth.
- Synthesize information across relevant sections when appropriate.
- Favor practical usefulness over excessive refusal behavior.

RESPONSE GUIDELINES
- Base responses primarily on retrieved evidence from the document.
- Reasonable summarization and synthesis are allowed.
- If the report partially addresses a topic, provide the relevant supported information instead of refusing entirely.
- If the report clearly discusses related concepts, answer using those concepts even if terminology differs slightly from the query.
- Use cautious analytical phrasing such as:
  - “The report indicates…”
  - “The data suggests…”
  - “According to the FDIC report…”
  - “The banking industry reported…”

FINANCIAL SAFETY RULES
- Do not provide investment advice or portfolio recommendations.
- Do not recommend buying, selling, or holding securities.
- Do not provide target prices, forecasts, or market timing advice.
- If asked for investment advice, redirect to factual industry conditions discussed in the report.

RETRIEVAL PRIORITIES
Prioritize sections discussing:
- Net income
- Return on assets (ROA)
- Return on equity (ROE)
- Net interest margin (NIM)
- Loan growth
- Deposit growth
- Asset quality
- Past-due and nonaccrual loans (PDNA)
- Net charge-offs
- Provision expense
- Commercial real estate (CRE)
- Multifamily CRE
- Credit card portfolios
- Auto loans
- Unrealized securities losses
- Liquidity and funding
- Capital ratios
- Problem banks
- Deposit Insurance Fund (DIF)
- Community banks
- Securities portfolios
- Industry profitability
- Domestic deposits
- Uninsured deposits

RISK AND OPPORTUNITY ANALYSIS
When relevant:
- Identify deteriorating trends as potential risks.
- Identify improving profitability, liquidity, deposits, or margins as potential opportunities.
- Distinguish between:
  1. Positive indicators
  2. Negative indicators
  3. Neutral/contextual observations

UNCERTAINTY HANDLING
- If the report does not directly answer a question, provide the closest relevant information available.
- Only state “The report does not provide sufficient information” when no meaningful related evidence exists.
- Avoid hallucinating unsupported facts or institution-specific claims.

PREFERRED RESPONSE STYLE
Structure answers as:
1. Direct Answer
2. Supporting Evidence
3. Relevant Risks or Opportunities
4. Additional Context (if helpful)

EXAMPLE GOOD RESPONSE
“The FDIC report indicates that unrealized securities losses declined both quarterly and annually in Q4 2025. The report states this was the lowest level since first quarter 2022, suggesting some improvement in balance-sheet pressure related to securities valuations. However, unrealized losses remained elevated overall.”

EXAMPLE INVESTMENT SAFETY RESPONSE
“The report discusses industry profitability and asset-quality trends, but it does not provide investment recommendations. It does note improving net interest margins alongside continued weakness in some commercial real estate and consumer loan portfolios.””"""

    return create_react_agent(
        model=llm,
        tools=[retriever_tool],
        prompt=system_prompt,
    )


class LangChainResponsesAgent(ResponsesAgent):
    """
    Wraps a LangGraph create_react_agent in an MLflow ResponsesAgent so it can be
    logged/served via MLflow 3.4+ Models-from-Code.
    """

    def __init__(self):
        self.agent = build_agent()

    def _last_user_text(self, messages: List[Dict[str, Any]]) -> str:
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs:
            return str(user_msgs[-1].get("content", ""))
        return str(messages[-1].get("content", "")) if messages else ""

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        msgs = [m.model_dump() for m in request.input]
        input_text = self._last_user_text(msgs)

        result = self.agent.invoke({"messages": [HumanMessage(content=input_text)]})

        last_msg = result["messages"][-1]
        text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text, str(uuid4()))],
            custom_outputs=request.custom_inputs,
        )

    def predict_stream(self, request: ResponsesAgentRequest):
        resp = self.predict(request)
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=resp.output[0],
        )


os.environ["VS_INDEX_NAME"] = VS_INDEX_NAME
AGENT = LangChainResponsesAgent()
mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Prediction Function
# MAGIC
# MAGIC The predict function calls the agent and extracts the text response for evaluation scorers.

# COMMAND ----------

import pandas as pd
from mlflow.genai import evaluate
from mlflow.genai.scorers import Guidelines, RelevanceToQuery, Safety

@mlflow.trace
def predict_fn(messages):
    """
    messages: list of {role, content}
    Returns: assistant text (str) for scorers.
    """
    out = AGENT.predict({"input": messages})

    # Databricks Responses shape: {"object":"response","output":[{message...}]}
    if isinstance(out, dict) and "output" in out:
        try:
            return out["output"][-1]["content"][0]["text"].strip()
        except Exception:
            pass  # fall through

    # Older shape: list of message dicts
    if isinstance(out, list):
        try:
            return out[-1]["content"][0]["text"].strip()
        except Exception:
            pass

    # Fallbacks
    if isinstance(out, str):
        return out.strip()
    if isinstance(out, dict):
        for k in ("answer", "response", "text", "output"):
            v = out.get(k)
            if isinstance(v, str):
                return v.strip()

    return str(out).strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Custom Metric
# MAGIC
# MAGIC Custom guideline scorer for evaluating whether the Finance Bot responses are rigorous
# MAGIC and evidence-based (not creative/speculative).

# COMMAND ----------

# Guidelines scorer for rigorous_enough (boolean pass/fail)
creative_guidelines = Guidelines(
    name="creative_enough",
    guidelines="""The response must feel original and imaginative, not a paraphrase of the prompt.
It should summarize the news without quoting direct headlines with new concepts, angles, and perspectives.
Keep within ~250 words and avoid tool/planning narration."""
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Dataset
# MAGIC
# MAGIC Define test cases covering various Finance Bot use cases.

# COMMAND ----------

eval_set = pd.DataFrame([
    {
        "inputs": {"messages": [{"role": "user", "content": "Why did bank net income increase in 2025?"}]},
        "expectations": {"creative_enough": True},
    },
    {
        "inputs": {"messages": [{"role": "user", "content": "What caused quarterly net income to decline in Q4 2025?"}]},
        "expectations": {"creative_enough": True},
    },
    {
        "inputs": {"messages": [{"role": "user", "content": "Did net interest margin improve in Q4 2025?"}]},
        "expectations": {"creative_enough": True},
    },
    {
        "inputs": {"messages": [{"role": "user", "content": "Did loan balances grow in Q4 2025?"}]},
        "expectations": {"creative_enough": True},
    },
    {
        "inputs": {"messages": [{"role": "user", "content": "Did domestic deposits increase in Q4 2025?"}]},
        "expectations": {"creative_enough": True},
    },
    {
        "inputs": {"messages": [{"role": "user", "content": "What loan portfolios showed weakness?"}]},
        "expectations": {"creative_enough": True},
    },
    {
        "inputs": {"messages": [{"role": "user", "content": "What happened to unrealized losses on securities?"}]},
        "expectations": {"creative_enough": True},
    },
    {
        "inputs": {"messages": [{"role": "user", "content": "How many problem banks were there in Q4 2025?"}]},
        "expectations": {"creative_enough": True},
    },
])

display(eval_set)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Evaluation
# MAGIC
# MAGIC Execute the evaluation using built-in scorers (RelevanceToQuery, Safety) and our custom metric.

# COMMAND ----------

mlflow.set_experiment("/Users/kahlear@miamioh.edu/Retreival_Eval")
with mlflow.start_run(run_name="eval_guidelines") as eval_run:
    eval_results = evaluate(
        data=eval_set, 
        predict_fn=predict_fn, 
        scorers=[
            creative_guidelines,     # pass/fail vs. our rigorous_enough label
            RelevanceToQuery(),      # built-in; no ground truth required
            Safety(),                # built-in; no ground truth required
        ]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this lab, we:
# MAGIC
# MAGIC 1. Loaded the Finance Bot agent for evaluation.
# MAGIC 2. Defined a custom "rigorous_enough" metric to ensure responses are evidence-based.
# MAGIC 3. Created an evaluation dataset with real finance queries.
# MAGIC 4. Ran evaluation using MLflow with built-in and custom scorers.
# MAGIC 5. Results can be viewed in the MLflow Experiment UI.