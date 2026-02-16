from collections.abc import Mapping
from typing import Any

from polars_nlq.definitions import Plan


SYSTEM_PROMPT = """You are a Polars query planner.
Return a JSON object that matches the provided Pydantic schema exactly.
Use only available columns from the given schema.
Prefer minimal plans and avoid unnecessary operations.
Avoid duplicate output names across all expressions.
When using groupby_agg, do not put the same grouping key in both by and named_by.
Use named_by only when you need to rename or derive grouping keys; otherwise keep named_by empty.
If aggregation is required, use groupby_agg with explicit aliases for metrics and unique output names.
Example for two-level grouping: for "sum of sales by country and city", use one groupby_agg with by=[col(country), col(city)] and aggs=[sum(col(sales)) as total_sales].
For row counts, use count() with no literal arguments (do not use lit(1) or lit(true)).
"""


def _format_schema(schema: Mapping[str, Any] | Any) -> str:
    if isinstance(schema, Mapping):
        items = schema.items()
    elif hasattr(schema, "items"):
        items = schema.items()
    else:
        raise TypeError("schema must be a mapping-like object")

    return "\n".join(f"- {name}: {dtype}" for name, dtype in items)


def nl_query(
    client: Any,
    schema: Mapping[str, Any] | Any,
    question: str,
    model: str = "local-model",
    system_prompt: str = SYSTEM_PROMPT,
) -> Plan:
    """Generate a typed query plan from natural language."""
    if not question.strip():
        raise ValueError("question must not be empty")

    schema_text = _format_schema(schema)

    create = getattr(getattr(getattr(client, "chat", None), "completions", None), "create", None)
    if create is None:
        raise TypeError("client must provide chat.completions.create")

    return create(
        model=model,
        response_model=Plan,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Generate a Plan for this question.\n"
                    f"Question: {question}\n"
                    "Available schema:\n"
                    f"{schema_text}"
                ),
            },
        ],
    )
