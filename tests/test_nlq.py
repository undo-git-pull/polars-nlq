import polars as pl

from polars_nlq import Col, Func, GroupByAgg, NamedExpr, Plan, execute_plan, nl_query


class _FakeCompletions:
    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self.called = False
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> Plan:
        self.called = True
        self.kwargs = kwargs
        return self._plan


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, plan: Plan) -> None:
        self.chat = _FakeChat(_FakeCompletions(plan))


def test_nl_query_creates_plan_from_schema(tmp_path) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("city,sales\nSeoul,30\nParis,50\nSeoul,20\n", encoding="utf-8")

    lf = pl.scan_csv(str(csv_path))

    expected_plan = Plan(
        ops=[
            GroupByAgg(
                by=[Col(name="city")],
                aggs=[NamedExpr(expr=Func(name="sum", args=[Col(name="sales")]), alias="total_sales")],
            )
        ]
    )
    client = _FakeClient(expected_plan)

    actual = nl_query(client, lf.collect_schema(), "sum of sales by city")

    assert actual == expected_plan
    assert client.chat.completions.called
    assert client.chat.completions.kwargs["response_model"] is Plan


def test_execute_plan_groupby_agg(tmp_path) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("city,sales\nSeoul,30\nParis,50\nSeoul,20\n", encoding="utf-8")

    lf = pl.scan_csv(str(csv_path))
    plan = Plan(
        ops=[
            GroupByAgg(
                by=[Col(name="city")],
                aggs=[NamedExpr(expr=Func(name="sum", args=[Col(name="sales")]), alias="total_sales")],
            )
        ]
    )

    result = execute_plan(lf, plan).collect().sort("city")

    assert result.to_dicts() == [
        {"city": "Paris", "total_sales": 50},
        {"city": "Seoul", "total_sales": 50},
    ]
