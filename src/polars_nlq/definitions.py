from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class UnaryOp(str, Enum):
    NOT = "not"
    NEG = "neg"


class BinaryOp(str, Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"

    EQ = "eq"
    NEQ = "neq"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"

    AND = "and"
    OR = "or"


class ExprBase(BaseModel):
    kind: str


class Col(ExprBase):
    kind: Literal["col"] = "col"
    name: str


class Lit(ExprBase):
    kind: Literal["lit"] = "lit"
    value: Any


class Unary(ExprBase):
    kind: Literal["unary"] = "unary"
    op: UnaryOp
    expr: "Expr"


class Binary(ExprBase):
    kind: Literal["binary"] = "binary"
    op: BinaryOp
    left: "Expr"
    right: "Expr"


class Func(ExprBase):
    kind: Literal["func"] = "func"
    name: str
    args: List["Expr"] = Field(default_factory=list)


class TopK(ExprBase):
    kind: Literal["topk"] = "topk"
    expr: "Expr"
    k: int = 5


class BottomK(ExprBase):
    kind: Literal["bottomk"] = "bottomk"
    expr: "Expr"
    k: int = 5


class WhenThenOtherwise(ExprBase):
    kind: Literal["when_then_otherwise"] = "when_then_otherwise"
    branches: List[tuple["Expr", "Expr"]] = Field(default_factory=list)
    otherwise: Optional["Expr"] = None


Expr = Union[Col, Lit, Unary, Binary, Func, TopK, BottomK, WhenThenOtherwise]


class NamedExpr(BaseModel):
    expr: Expr
    alias: Optional[str] = None


class SortKey(BaseModel):
    expr: Expr
    descending: bool = False
    nulls_last: bool = True


class OpBase(BaseModel):
    op: str


class Select(OpBase):
    op: Literal["select"] = "select"
    exprs: List[NamedExpr]


class WithColumns(OpBase):
    op: Literal["with_columns"] = "with_columns"
    exprs: List[NamedExpr]


class Filter(OpBase):
    op: Literal["filter"] = "filter"
    predicate: Expr


class Sort(OpBase):
    op: Literal["sort"] = "sort"
    by: List[SortKey]


class Limit(OpBase):
    op: Literal["limit"] = "limit"
    n: int


class GroupByAgg(OpBase):
    op: Literal["groupby_agg"] = "groupby_agg"
    by: List[Expr] = Field(default_factory=list)
    maintain_order: bool = False
    named_by: Dict[str, Expr] = Field(default_factory=dict)
    aggs: List[NamedExpr] = Field(default_factory=list)
    named_aggs: Dict[str, Expr] = Field(default_factory=dict)


Op = Union[Select, WithColumns, Filter, GroupByAgg, Sort, Limit]


class Plan(BaseModel):
    ops: List[Op] = Field(default_factory=list)


Unary.model_rebuild()
Binary.model_rebuild()
Func.model_rebuild()
TopK.model_rebuild()
BottomK.model_rebuild()
WhenThenOtherwise.model_rebuild()
