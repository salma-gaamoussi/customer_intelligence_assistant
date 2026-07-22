import os
import re
from typing import Any, Optional

from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools import BaseTool, QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

from app.guardrails import check_sql_is_select_only, enforce_row_limit

load_dotenv()

MAX_RESULT_ROWS = 50
MAX_AGENT_ITERATIONS = 5
SQL_QUERY_TOOL_NAME = "sql_db_query"


class SQLAgentResult(BaseModel):
    answer: str
    sql_queries: list[str]


def get_readonly_database_url() -> str:
    db_url = os.getenv("TELCO_READONLY_DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "TELCO_READONLY_DATABASE_URL is not set."
        )
    return db_url


def build_sql_database() -> SQLDatabase:
    return SQLDatabase.from_uri(
        get_readonly_database_url(),
        schema="telco",
        include_tables=["customers"],
        sample_rows_in_table_info=2,
    )


def _cap_query_rows(query: str, max_rows: int) -> str:
    stripped = query.strip().rstrip(";")
    if not re.match(r"(?is)^\s*select\b", stripped):
        return stripped
    if re.search(r"(?is)\blimit\s+\d+\s*$", stripped):
        return stripped
    return f"{stripped} LIMIT {max_rows}"


class RowCappedQuerySQLDatabaseTool(QuerySQLDatabaseTool):

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        capped_query = _cap_query_rows(query, MAX_RESULT_ROWS)

        sql_check = check_sql_is_select_only(capped_query)
        if not sql_check.allowed:
            return f"Error: {sql_check.reason}"

        result = super()._run(capped_query, run_manager=run_manager)
        if isinstance(result, str):
            result = enforce_row_limit(result, MAX_RESULT_ROWS)
        return result


class RowCappedSQLDatabaseToolkit(SQLDatabaseToolkit):
    def get_tools(self) -> list[BaseTool]:
        tools = super().get_tools()
        for i, tool in enumerate(tools):
            if tool.name == SQL_QUERY_TOOL_NAME:
                tools[i] = RowCappedQuerySQLDatabaseTool(db=self.db, description=tool.description)
        return tools


def build_sql_agent(db: SQLDatabase) -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    toolkit = RowCappedSQLDatabaseToolkit(db=db, llm=llm)

    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="tool-calling",
        top_k=MAX_RESULT_ROWS,
        max_iterations=MAX_AGENT_ITERATIONS,
        agent_executor_kwargs={"return_intermediate_steps": True},
    )


def extract_executed_sql(intermediate_steps: list) -> list[str]:
    queries: list[str] = []
    for action, _observation in intermediate_steps:
        if action.tool != SQL_QUERY_TOOL_NAME:
            continue
        tool_input = action.tool_input
        query = tool_input.get("query", "") if isinstance(tool_input, dict) else str(tool_input)
        if not query:
            continue
        capped_query = _cap_query_rows(query, MAX_RESULT_ROWS)
        if capped_query not in queries:
            queries.append(capped_query)
    return queries


def ask_sql_agent(question: str) -> SQLAgentResult:
    db = build_sql_database()
    agent_executor = build_sql_agent(db)

    try:
        result = agent_executor.invoke({"input": question})
    except Exception as exc:
        return SQLAgentResult(
            answer=f"I couldn't answer that from the customer database ({exc}).",
            sql_queries=[],
        )

    sql_queries = extract_executed_sql(result.get("intermediate_steps", []))
    output = result.get("output", "")

    if "iteration limit" in output.lower() or "time limit" in output.lower():
        return SQLAgentResult(
            answer="I wasn't able to find a reliable answer to that within the allowed steps.",
            sql_queries=sql_queries,
        )

    return SQLAgentResult(answer=output, sql_queries=sql_queries)
