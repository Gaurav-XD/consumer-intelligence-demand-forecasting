"""Typed record definitions for the project's input datasets."""

from datetime import date
from decimal import Decimal
from typing import TypedDict


class CustomerRecord(TypedDict):
    """Expected fields for a customer record."""

    customer_id: str
    age: int
    gender: str
    region: str
    signup_date: date


class TransactionRecord(TypedDict):
    """Expected fields for a transaction record."""

    transaction_id: str
    customer_id: str
    transaction_date: date
    product_id: str
    quantity: int
    price: Decimal
    discount: Decimal


class ProductRecord(TypedDict):
    """Expected fields for a product record."""

    product_id: str
    category: str
    brand: str
    cost: Decimal


class MarketingRecord(TypedDict):
    """Expected fields for a marketing record."""

    customer_id: str
    campaign: str
    campaign_date: date
    channel: str
