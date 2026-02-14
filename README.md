## polars-nlq

`polars-nlq` turns natural language questions into a typed Polars query plan, then executes that plan against a `DataFrame` or `LazyFrame`.

The library is built around two functions:

- `nl_query(...)` to generate a validated `Plan` with `instructor`
- `execute_plan(...)` to execute that `Plan` with Polars

Warning: plans created by LLMs can be incorrect and should be reviewed by a human before use.

## Install

```bash
uv sync
```

## Quick Example

```python
import instructor
import polars as pl
from openai import OpenAI

from polars_nlq import execute_plan, nl_query

# Columns: name,city,sales
q1 = pl.scan_csv("sales.csv")

# Local OpenAI-compatible endpoint
openai_client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")
client = instructor.from_openai(openai_client)

plan = nl_query(client, q1.collect_schema(), "sum of sales by city, with at least 20 sales")

results_lf = execute_plan(q1, plan)  # execute_plan always returns a LazyFrame

print(results_lf.collect())
```

## Plan Model

Plans are typed with Pydantic models and validated before execution.

- Expressions: `col`, `lit`, `unary`, `binary`, `func`, `when_then_otherwise`
- Operations: `select`, `with_columns`, `filter`, `groupby_agg`, `sort`, `limit`

This gives you a clear contract between LLM output and execution, and plans can be serialized and reused.

Example plan from the query above:
```
ops=[GroupByAgg(op='groupby_agg', by=[Col(kind='col', name='city')], maintain_order=False, named_by={}, aggs=[NamedExpr(expr=Func(kind='func', name='sum', args=[Col(kind='col', name='sales')]), alias='total_sales')], named_aggs={}), Filter(op='filter', predicate=Binary(kind='binary', op=<BinaryOp.GTE: 'gte'>, left=Col(kind='col', name='total_sales'), right=Lit(kind='lit', value=20))), Select(op='select', exprs=[NamedExpr(expr=Col(kind='col', name='city'), alias=None), NamedExpr(expr=Col(kind='col', name='total_sales'), alias=None)])]
```

## API

### `nl_query(client, schema, question, model="local-model") -> Plan`

- `client`: instructor-wrapped client that supports `chat.completions.create`
- `schema`: mapping-like schema (for example `LazyFrame.collect_schema()`)
- `question`: natural language prompt
- `model`: model name passed to `chat.completions.create` (defaults to `"local-model"`)

Returns a validated `Plan`.

### `execute_plan(source, plan) -> pl.LazyFrame`

- `source`: `pl.DataFrame` or `pl.LazyFrame`
- `plan`: `Plan` instance or compatible dict

Returns collected query results as a `pl.LazyFrame`.

## Run Tests

```bash
uv run pytest -q
```

## Limitations

- Derived columns are not supported in plans (for example, grouping by year from a date column).
- Joins are not supported.
