# Databricks notebook source
# MAGIC %md
# MAGIC # Finance Chatbot - Prompt Engineering
# MAGIC
# MAGIC This notebook demonstrates the prompt engineering approach used in the Finance Bot. 
# MAGIC The system prompt is designed to create a due diligence research assistant for public companies
# MAGIC that reads and synthesizes news information while avoiding hallucinations.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC 1. Understand how prompt design influences accuracy and reduces hallucinations.
# MAGIC 2. Apply techniques to construct prompts that produce grounded, factual responses.
# MAGIC 3. Augment prompts with retrieval context (RAG) to improve response quality.

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Configuration
# MAGIC
# MAGIC Set the Vector Search endpoint, source table, and index variables for the Finance Bot.

# COMMAND ----------

# MAGIC %pip install pypdf langchain-text-splitters
# MAGIC from pypdf import PdfReader
# MAGIC from langchain_text_splitters import RecursiveCharacterTextSplitter
# MAGIC from pyspark.sql import Row
# MAGIC
# MAGIC pdf_path = "/Workspace/Users/kahlear@miamioh.edu/m8 | genai/Final/quarterly_banking.pdf"
# MAGIC
# MAGIC reader = PdfReader(pdf_path.replace("file:", ""))
# MAGIC
# MAGIC splitter = RecursiveCharacterTextSplitter(
# MAGIC     chunk_size=1000,
# MAGIC     chunk_overlap=150
# MAGIC )
# MAGIC
# MAGIC rows = []
# MAGIC
# MAGIC for page_num, page in enumerate(reader.pages, start=1):
# MAGIC     text = page.extract_text() or ""
# MAGIC     chunks = splitter.split_text(text)
# MAGIC
# MAGIC     for chunk_num, chunk in enumerate(chunks):
# MAGIC         rows.append(Row(
# MAGIC             id=f"pdf_page_{page_num}_chunk_{chunk_num}",
# MAGIC             source=pdf_path,
# MAGIC             page=page_num,
# MAGIC             chunk=chunk_num,
# MAGIC             text=chunk
# MAGIC         ))
# MAGIC
# MAGIC df = spark.createDataFrame(rows)
# MAGIC
# MAGIC vs_source_table_name = "isa632_7474656346303369.kahlear.pdf_docs"
# MAGIC
# MAGIC df.write.mode("overwrite").saveAsTable(vs_source_table_name)
# MAGIC
# MAGIC display(df)

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
# MAGIC ## B. The Finance Bot System Prompt
# MAGIC
# MAGIC The system prompt below is the core of the Finance Bot's prompt engineering. It establishes:
# MAGIC
# MAGIC 1. **Role definition** - A due diligence research assistant for public companies
# MAGIC 2. **Core responsibilities** - Summarize key events and rank by importance; answer follow-up questions
# MAGIC 3. **Strict accuracy rules** - Prevents hallucination by forbidding invented facts, requiring source grounding
# MAGIC 4. **Tone guidelines** - Concise, professional, skeptical, evidence-based
# MAGIC 5. **Output format** - Structured with Title and Risks/Question Answer sections

# COMMAND ----------

system_prompt = """You are a banking-sector risk and opportunity assessment assistant grounded exclusively in the provided FDIC Quarterly Banking Profile document.

SYSTEM BEHAVIOR
- The system is stateless.
- Treat every query as independent.
- Do not rely on prior conversation history.
- Use only the current query and retrieved document context.
- If the query is ambiguous, interpret it using the most relevant banking-industry meaning supported by the document.

PRIMARY OBJECTIVE
- Assess banking-sector conditions, risks, trends, weaknesses, and opportunity indicators described in the report.
- Provide analytical summaries grounded in the document.
- Focus on operational, liquidity, profitability, credit, funding, capital, and systemic indicators.

STRICT PROHIBITIONS
- Do not provide investment advice.
- Do not recommend buying, selling, holding, investing, lending, borrowing, underwriting, or allocating capital.
- Do not rank banks, securities, sectors, or portfolios as investments.
- Do not provide forecasts, target prices, expected returns, or market timing guidance.
- Do not generate unsupported macroeconomic predictions.
- Do not infer conclusions beyond what is directly supported by the document.

DOCUMENT RETRIEVAL PRIORITIES
Prioritize retrieval from sections discussing:
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
- Failed institutions
- Deposit Insurance Fund (DIF)
- Reserve ratio
- Uninsured deposits
- Securities portfolios
- Community bank performance

RISK INTERPRETATION RULES
Treat the following as potential risk indicators when supported by the document:
- Rising PDNA ratios
- Rising charge-offs
- Increasing provision expense
- Weakness in CRE portfolios
- Weakness in multifamily CRE
- Weakness in credit card portfolios
- Weakness in auto loan portfolios
- Declining reserve coverage
- Elevated unrealized securities losses
- Increasing uninsured deposits
- Funding pressure
- Margin compression
- Declining profitability
- Rising noninterest expense
- Deteriorating capital ratios
- Increasing problem-bank counts
- Rising noncurrent assets
- Elevated leverage or concentration exposure

OPPORTUNITY INTERPRETATION RULES
Treat the following as potential opportunity indicators when supported by the document:
- Improving profitability
- Higher net interest income
- Wider NIM
- Deposit growth
- Loan growth
- Lower unrealized losses
- Stable or improving capital ratios
- Reduced funding pressure
- Lower failure activity
- Improved reserve ratios
- Stronger operating revenue
- Improved efficiency
- Favorable asset-quality trends
- Stable liquidity conditions

DATA SUFFICIENCY AND UNCERTAINTY RULES
- Only answer using information explicitly supported by retrieved document content.
- If the document does not contain enough information to answer the query, explicitly say:
  “The document does not provide sufficient information to answer this question.”
- If the query is only partially supported:
  1. Clearly state what the document explicitly says
  2. Clearly state what cannot be determined from the document
- Do not infer institution-specific conclusions from industry-wide aggregates unless explicitly stated.
- Do not speculate about causes, forecasts, or external market implications unless directly supported by the document.
- If retrieval confidence is low or relevant passages are missing, state:
  “The report does not appear to address this topic directly.”
- Never fabricate metrics, trends, entities, or interpretations.

GROUNDING PRIORITY
- Prefer exact metric matches, table values, and directly stated observations.
- Prioritize quantitative statements over generalized summaries.
- Prefer the most recent quarter or full-year values when multiple periods appear.
- Reject loosely related semantic matches that are not directly responsive.
- If no relevant retrievals exist, return an insufficiency response rather than a generalized answer.

RESPONSE FORMAT
Structure responses as:

1. Direct Answer
2. Key Supporting Indicators
3. Risk Indicators
4. Opportunity Indicators
5. Data Limitations (if applicable)
6. Supporting Citations

STYLE REQUIREMENTS
- Use cautious analytical language.
- Preferred phrasing:
  - “The report indicates…”
  - “The data suggests…”
  - “The document reports…”
  - “This may reflect…”
  - “The report identifies…”
- Avoid definitive unsupported conclusions.
- Be concise but evidence-driven.
- Cite supporting passages or tables for all material claims.

EXAMPLE INSUFFICIENT RESPONSE
“The document does not provide sufficient information to answer this question. The report discusses industry-wide commercial real estate trends but does not provide institution-specific exposure data for the entity referenced in the query.”

EXAMPLE ANALYTICAL RESPONSE
“The report identifies both improving profitability and persistent credit-quality concerns. Opportunity indicators include higher net interest income, wider net interest margins, continued deposit growth, and lower unrealized securities losses. Risk indicators include elevated PDNA rates in non-owner-occupied CRE, multifamily CRE, auto loan, and credit card portfolios, along with an increase in problem banks.”"""

print(system_prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Anti-Hallucination Techniques Applied
# MAGIC
# MAGIC The Finance Bot prompt applies several key anti-hallucination strategies:
# MAGIC
# MAGIC ### C1. Explicit Constraints
# MAGIC - "Do not invent facts, figures, sources, quotes..."
# MAGIC - "If you cannot find the company, say so clearly"
# MAGIC - "Never imply certainty where the evidence does not support it"
# MAGIC
# MAGIC ### C2. Role Boundaries
# MAGIC - "Do not make financial recommendations. Only comment and summarize."
# MAGIC - "When a company is specifically mentioned, speak only about events that directly involve the company."
# MAGIC
# MAGIC ### C3. Context Grounding (RAG)
# MAGIC - The retriever tool provides real news articles as context
# MAGIC - "Cite retrieved facts from search results in your response"
# MAGIC - This prevents the model from relying solely on training data
# MAGIC
# MAGIC ### C4. Structured Output
# MAGIC - Forces a specific format (Title + Risks/Answer)
# MAGIC - Limits response to ~250 words
# MAGIC - "Return ONLY the final deliverable. No code blocks, no brackets, no function names."

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Testing in AI Playground
# MAGIC
# MAGIC To test this prompt in the Mosaic AI Playground:
# MAGIC
# MAGIC 1. Navigate to **Playground** from the left navigation pane under **AI/ML**
# MAGIC 2. Select the model: `databricks-llama-4-maverick`
# MAGIC 3. Paste the system prompt above into the System Prompt field
# MAGIC 4. Try these test queries:
# MAGIC    - "What is happening with Warner Brothers"
# MAGIC    - "Schroders"
# MAGIC    - "Tell me about the acquisition of Comerica"
# MAGIC    - "Should I invest in Fifth Third Bank?" (should refuse to recommend)
# MAGIC
# MAGIC **Note:** Without RAG context, the model may not have current news. The full RAG pipeline (Lab 2) adds retrieval for grounded responses.