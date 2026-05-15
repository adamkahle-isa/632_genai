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
“The report discusses industry profitability and asset-quality trends, but it does not provide investment recommendations. It does note improving net interest margins alongside continued weakness in some commercial real estate and consumer loan portfolios."""

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

        # LangGraph returns a messages list; grab the last AIMessage
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


# Register model object for MLflow Models-from-Code
AGENT = LangChainResponsesAgent()
mlflow.models.set_model(AGENT)
