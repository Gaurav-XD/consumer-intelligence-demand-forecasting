"""Strict request and response contracts for model inference."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonnegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
NonnegativeInt = Annotated[int, Field(ge=0, strict=True)]
OptionalFloat = NonnegativeFloat | None
Discount = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)] | None


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChurnFeatures(StrictSchema):
    customer_tenure_days: NonnegativeInt
    total_orders: NonnegativeInt
    total_quantity: NonnegativeInt
    total_revenue: NonnegativeFloat
    average_order_value: OptionalFloat
    average_discount: Discount
    unique_products_purchased: NonnegativeInt
    unique_categories_purchased: NonnegativeInt
    purchase_frequency: OptionalFloat
    average_days_between_purchases: OptionalFloat
    days_since_last_purchase: NonnegativeInt | None
    customer_lifetime_days: NonnegativeInt | None
    number_marketing_interactions: NonnegativeInt
    number_marketing_campaigns: NonnegativeInt


class ForecastFeatures(StrictSchema):
    lag_1_quantity: NonnegativeInt
    lag_2_quantity: NonnegativeInt
    lag_3_quantity: NonnegativeInt
    lag_6_quantity: NonnegativeInt
    lag_12_quantity: NonnegativeInt
    rolling_3_mean_quantity: NonnegativeFloat
    rolling_6_mean_quantity: NonnegativeFloat
    rolling_12_mean_quantity: NonnegativeFloat
    rolling_3_std_quantity: NonnegativeFloat
    rolling_6_std_quantity: NonnegativeFloat
    month_of_year: Annotated[int, Field(ge=1, le=12, strict=True)]
    quarter: Annotated[int, Field(ge=1, le=4, strict=True)]
    trend_index: Annotated[int, Field(ge=12, le=119999, strict=True)]

    @model_validator(mode="after")
    def calendar_consistency(self) -> "ForecastFeatures":
        if self.quarter != (self.month_of_year - 1) // 3 + 1:
            raise ValueError("quarter must match month_of_year")
        if self.trend_index % 12 != self.month_of_year - 1:
            raise ValueError("trend_index must match month_of_year")
        return self


class ModelMetadata(StrictSchema):
    name: str
    version: str
    artifact_sha256: str
    trained_scikit_learn_version: str
    feature_columns: list[str]


class HealthResponse(StrictSchema):
    status: Literal["ok"]


class ModelListResponse(StrictSchema):
    models: dict[str, ModelMetadata]


class ChurnResponse(StrictSchema):
    churn_probability: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    predicted_class: Literal[0, 1]
    model: ModelMetadata


class ForecastResponse(StrictSchema):
    predicted_quantity: Annotated[float, Field(allow_inf_nan=False)]
    model: ModelMetadata


class ErrorResponse(StrictSchema):
    error: str
    message: str
