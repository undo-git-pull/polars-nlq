from typing import Any, Mapping

import polars as pl

from polars_nlq.definitions import (
    Binary,
    BinaryOp,
    Col,
    Expr,
    Filter,
    Func,
    GroupByAgg,
    Limit,
    Lit,
    NamedExpr,
    Plan,
    Select,
    Sort,
    Unary,
    UnaryOp,
    WhenThenOtherwise,
    WithColumns,
)


def _to_named_expr(node: NamedExpr) -> pl.Expr:
    expr = _to_expr(node.expr)
    if node.alias:
        return expr.alias(node.alias)
    return expr


def _to_expr(expr: Expr) -> pl.Expr:
    if isinstance(expr, Col):
        return pl.col(expr.name)
    if isinstance(expr, Lit):
        return pl.lit(expr.value)
    if isinstance(expr, Unary):
        inner = _to_expr(expr.expr)
        if expr.op == UnaryOp.NOT:
            return ~inner
        if expr.op == UnaryOp.NEG:
            return -inner
        raise ValueError(f"Unsupported unary operation: {expr.op}")
    if isinstance(expr, Binary):
        left = _to_expr(expr.left)
        right = _to_expr(expr.right)
        if expr.op == BinaryOp.ADD:
            return left + right
        if expr.op == BinaryOp.SUB:
            return left - right
        if expr.op == BinaryOp.MUL:
            return left * right
        if expr.op == BinaryOp.DIV:
            return left / right
        if expr.op == BinaryOp.EQ:
            return left == right
        if expr.op == BinaryOp.NEQ:
            return left != right
        if expr.op == BinaryOp.LT:
            return left < right
        if expr.op == BinaryOp.LTE:
            return left <= right
        if expr.op == BinaryOp.GT:
            return left > right
        if expr.op == BinaryOp.GTE:
            return left >= right
        if expr.op == BinaryOp.AND:
            return left & right
        if expr.op == BinaryOp.OR:
            return left | right
        raise ValueError(f"Unsupported binary operation: {expr.op}")
    if isinstance(expr, Func):
        args = [_to_expr(arg) for arg in expr.args]
        name = expr.name.lower().strip()
        return _apply_func(name, args)
    if isinstance(expr, WhenThenOtherwise):
        if not expr.branches:
            raise ValueError("when_then_otherwise requires at least one branch")
        when, then = expr.branches[0]
        chain = pl.when(_to_expr(when)).then(_to_expr(then))
        for when, then in expr.branches[1:]:
            chain = chain.when(_to_expr(when)).then(_to_expr(then))
        if expr.otherwise is None:
            return chain.otherwise(None)
        return chain.otherwise(_to_expr(expr.otherwise))
    raise TypeError(f"Unsupported expression type: {type(expr)!r}")


def _apply_func(name: str, args: list[pl.Expr]) -> pl.Expr:
    if name == "sum":
        return args[0].sum()
    if name in {"mean", "avg"}:
        return args[0].mean()
    if name == "min":
        return args[0].min()
    if name == "max":
        return args[0].max()
    if name == "count":
        if args:
            return args[0].count()
        return pl.len()
    if name == "n_unique":
        return args[0].n_unique()
    if name == "first":
        return args[0].first()
    if name == "last":
        return args[0].last()
    if name == "abs":
        return args[0].abs()
    if name == "round":
        if len(args) > 1:
            raise ValueError("round supports a single argument in this plan engine")
        return args[0].round(0)
    if name == "lower":
        return args[0].str.to_lowercase()
    if name == "upper":
        return args[0].str.to_uppercase()
    raise ValueError(f"Unsupported function: {name}")


def execute_plan(source: pl.LazyFrame | pl.DataFrame, plan: Plan | Mapping[str, Any]) -> pl.LazyFrame:
    """Execute a plan and return a LazyFrame."""
    plan_obj = Plan.model_validate(plan)

    lf = source.lazy() if isinstance(source, pl.DataFrame) else source

    for op in plan_obj.ops:
        if isinstance(op, Select):
            lf = lf.select([_to_named_expr(item) for item in op.exprs])
        elif isinstance(op, WithColumns):
            lf = lf.with_columns([_to_named_expr(item) for item in op.exprs])
        elif isinstance(op, Filter):
            lf = lf.filter(_to_expr(op.predicate))
        elif isinstance(op, GroupByAgg):
            by = [_to_expr(item) for item in op.by]
            named_by = {k: _to_expr(v) for k, v in op.named_by.items()}
            aggs = [_to_named_expr(item) for item in op.aggs]
            named_aggs = {k: _to_expr(v) for k, v in op.named_aggs.items()}
            lf = lf.group_by(*by, maintain_order=op.maintain_order, **named_by).agg(*aggs, **named_aggs)
        elif isinstance(op, Sort):
            by = [_to_expr(key.expr) for key in op.by]
            descending = [key.descending for key in op.by]
            nulls_last = [key.nulls_last for key in op.by]
            lf = lf.sort(by=by, descending=descending, nulls_last=nulls_last)
        elif isinstance(op, Limit):
            lf = lf.limit(op.n)
        else:
            raise ValueError(f"Unsupported operation: {op}")

    return lf
