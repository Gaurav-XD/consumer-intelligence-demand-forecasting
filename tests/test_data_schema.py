"""Tests for the importable dataset schema definitions."""

import unittest
from datetime import date
from decimal import Decimal
from typing import get_type_hints

from consumer_intelligence.data_schema import (
    CustomerRecord,
    MarketingRecord,
    ProductRecord,
    TransactionRecord,
)


class DataSchemaTests(unittest.TestCase):
    def test_customer_schema_fields_and_types(self) -> None:
        self.assertEqual(
            get_type_hints(CustomerRecord),
            {
                "customer_id": str,
                "age": int,
                "gender": str,
                "region": str,
                "signup_date": date,
            },
        )

    def test_transaction_schema_fields_and_types(self) -> None:
        self.assertEqual(
            get_type_hints(TransactionRecord),
            {
                "transaction_id": str,
                "customer_id": str,
                "transaction_date": date,
                "product_id": str,
                "quantity": int,
                "price": Decimal,
                "discount": Decimal,
            },
        )

    def test_product_schema_fields_and_types(self) -> None:
        self.assertEqual(
            get_type_hints(ProductRecord),
            {
                "product_id": str,
                "category": str,
                "brand": str,
                "cost": Decimal,
            },
        )

    def test_marketing_schema_fields_and_types(self) -> None:
        self.assertEqual(
            get_type_hints(MarketingRecord),
            {
                "customer_id": str,
                "campaign": str,
                "campaign_date": date,
                "channel": str,
            },
        )


if __name__ == "__main__":
    unittest.main()
