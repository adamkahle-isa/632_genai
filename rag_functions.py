from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.getActiveSession()

# -----------------------------
# NEWS RETRIEVAL
# -----------------------------
def get_articles(company_tag: str, query_type: str = None, limit: int = 5):
    query = f"""
        SELECT title, body, source, published_at, query_type
        FROM news_articles
        WHERE lower(company_tag) LIKE lower('%{company_tag}%')
    """

    if query_type:
        query += f" AND lower(query_type) = '{query_type.lower()}'"

    query += f" ORDER BY published_at DESC LIMIT {limit}"

    results = spark.sql(query).collect()

    return [row.asDict() for row in results]


# -----------------------------
# COUNT ARTICLES
# -----------------------------
def count_articles(company_tag: str):
    query = f"""
        SELECT COUNT(*) as count
        FROM news_articles
        WHERE lower(company_tag) LIKE lower('%{company_tag}%')
    """

    result = spark.sql(query).collect()[0]["count"]
    return result


# -----------------------------
# COMPANY INFO
# -----------------------------
def get_company_info(company_tag: str):
    try:
        query = f"""
            SELECT *
            FROM company_data
            WHERE lower(company_tag) LIKE lower('%{company_tag}%')
            LIMIT 1
        """
        result = spark.sql(query).collect()
        return result[0].asDict() if result else {}
    except:
        return {}


# -----------------------------
# LLM STUB
# -----------------------------
def call_llm(prompt: str) -> str:
    return f"[LLM OUTPUT PLACEHOLDER]\n\n{prompt[:500]}"


# -----------------------------
# SUMMARIZATION
# -----------------------------
def summarize_company(company_tag: str, query_type: str = None):
    articles = get_articles(company_tag, query_type=query_type)
    company_info = get_company_info(company_tag)

    article_text = "\n\n".join(
        [f"{a['title']}:\n{a['body'][:300]}" for a in articles]
    )

    prompt = f"""
    Summarize recent financial activity for {company_tag}.

    Company Info:
    {company_info}

    Articles:
    {article_text}
    """

    return call_llm(prompt)


# -----------------------------
# MAIN AGENT LOGIC
# -----------------------------
def agent(query: str):
    q = query.lower()

    if "warner" in q:
        company = "Warner Bros"
    elif "webster" in q:
        company = "Webster Financial"
    else:
        return "Company not recognized yet."

    if "m&a" in q or "acquisition" in q:
        return summarize_company(company, query_type="mna")

    if "summarize" in q:
        return summarize_company(company)

    if "how many" in q or "count" in q:
        return f"Total articles: {count_articles(company)}"

    return "Query not supported yet."
