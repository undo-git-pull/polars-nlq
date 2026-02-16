import polars as pl

from polars_nlq import BottomK, Col, Func, GroupByAgg, NamedExpr, Plan, Select, TopK, execute_plan, nl_query


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


def test_execute_plan_topk_expr() -> None:
    df = pl.DataFrame({"sales": [30, 50, 20, 10]})
    plan = Plan(ops=[Select(exprs=[NamedExpr(expr=TopK(expr=Col(name="sales"), k=2), alias="top_sales")])])

    result = execute_plan(df, plan).collect()

    assert result.to_dicts() == [{"top_sales": 50}, {"top_sales": 30}]


def test_execute_plan_bottomk_expr() -> None:
    df = pl.DataFrame({"sales": [30, 50, 20, 10]})
    plan = Plan(ops=[Select(exprs=[NamedExpr(expr=BottomK(expr=Col(name="sales"), k=2), alias="bottom_sales")])])

    result = execute_plan(df, plan).collect()

    assert result.to_dicts() == [{"bottom_sales": 10}, {"bottom_sales": 20}]


def test_execute_plan_top_2_cities_by_country() -> None:
    df = pl.DataFrame(
        {
            "country": ["France", "France", "France", "Japan", "Japan", "Japan"],
            "city": ["Paris", "Lyon", "Marseille", "Tokyo", "Osaka", "Nagoya"],
            "sales": [50, 35, 15, 10, 40, 22],
        }
    )
    plan = Plan(
        ops=[
            GroupByAgg(
                by=[Col(name="country"), Col(name="city")],
                aggs=[NamedExpr(expr=Func(name="sum", args=[Col(name="sales")]), alias="total_sales")],
            ),
            GroupByAgg(
                by=[Col(name="country")],
                aggs=[NamedExpr(expr=TopK(expr=Col(name="total_sales"), k=2), alias="top2_city_sales")],
            ),
        ]
    )

    result = execute_plan(df, plan).collect().sort("country")
    actual = [
        {"country": row["country"], "top2_city_sales": sorted(row["top2_city_sales"], reverse=True)}
        for row in result.to_dicts()
    ]

    assert actual == [
        {"country": "France", "top2_city_sales": [50, 35]},
        {"country": "Japan", "top2_city_sales": [40, 22]},
    ]


def test_execute_plan_bottom_1_city_by_country() -> None:
    df = pl.DataFrame(
        {
            "country": ["France", "France", "France", "Japan", "Japan", "Japan"],
            "city": ["Paris", "Lyon", "Marseille", "Tokyo", "Osaka", "Nagoya"],
            "sales": [50, 35, 15, 10, 40, 22],
        }
    )
    plan = Plan(
        ops=[
            GroupByAgg(
                by=[Col(name="country"), Col(name="city")],
                aggs=[NamedExpr(expr=Func(name="sum", args=[Col(name="sales")]), alias="total_sales")],
            ),
            GroupByAgg(
                by=[Col(name="country")],
                aggs=[NamedExpr(expr=BottomK(expr=Col(name="total_sales"), k=1), alias="bottom1_city_sales")],
            ),
        ]
    )

    result = execute_plan(df, plan).collect().sort("country")

    assert result.to_dicts() == [
        {"country": "France", "bottom1_city_sales": [15]},
        {"country": "Japan", "bottom1_city_sales": [10]},
    ]
