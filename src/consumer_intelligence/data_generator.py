"""Generate reproducible synthetic retail datasets for local development."""

import argparse
import csv
import random
from bisect import bisect_right
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TypedDict

from consumer_intelligence.config import RAW_DATA_DIR
from consumer_intelligence.data_schema import (
    CustomerRecord,
    MarketingRecord,
    ProductRecord,
    TransactionRecord,
)

DEFAULT_SEED = 42
DEFAULT_CUSTOMER_COUNT = 5_000
DEFAULT_PRODUCT_COUNT = 100
DEFAULT_TRANSACTION_COUNT = 60_000
DEFAULT_START_DATE = date(2024, 1, 1)
DEFAULT_END_DATE = date(2025, 12, 31)

CATEGORIES = {
    "Apparel": (8.0, 55.0, 2.4),
    "Beauty": (3.0, 35.0, 3.0),
    "Electronics": (25.0, 450.0, 1.65),
    "Grocery": (1.0, 25.0, 2.0),
    "Home": (6.0, 180.0, 2.2),
    "Sports": (7.0, 160.0, 2.1),
}
REGIONS = ("Northeast", "Midwest", "South", "West", "Central")
BRANDS = (
    "Northstar",
    "Everwell",
    "Field & Form",
    "Juniper Lane",
    "Brightline",
    "Common Ground",
    "Redwood",
    "Bluebird",
    "Summit",
    "Everyday Co.",
)
CAMPAIGNS = (
    "Welcome Series",
    "Loyalty Rewards",
    "Spring Sale",
    "Summer Sale",
    "Back-to-School",
    "Holiday Promotion",
    "Win-Back",
)
CHANNELS = ("Email", "SMS", "Paid Social", "Direct Mail", "Push Notification")

DATASET_FIELDS = {
    "customers": tuple(CustomerRecord.__annotations__),
    "products": tuple(ProductRecord.__annotations__),
    "transactions": tuple(TransactionRecord.__annotations__),
    "marketing": tuple(MarketingRecord.__annotations__),
}


class DatasetCollection(TypedDict):
    customers: list[CustomerRecord]
    products: list[ProductRecord]
    transactions: list[TransactionRecord]
    marketing: list[MarketingRecord]


class _CustomerProfile(TypedDict):
    behavior: str
    intensity: float
    favorite_category: str
    decline_start_month: int
    inactive_start_month: int


def _cumulative_weights(weights: list[float]) -> list[float]:
    cumulative: list[float] = []
    running_total = 0.0
    for weight in weights:
        running_total += weight
        cumulative.append(running_total)
    return cumulative


def _weighted_index(rng: random.Random, cumulative: list[float]) -> int:
    return bisect_right(cumulative, rng.random() * cumulative[-1])


def _month_index(day: date, start: date) -> int:
    return (day.year - start.year) * 12 + day.month - start.month


def _day_weights(start: date, end: date) -> tuple[list[date], list[float]]:
    days: list[date] = []
    weights: list[float] = []
    current = start
    while current <= end:
        weight = 1.0
        if current.weekday() >= 5:
            weight *= 1.12
        if current.month in (11, 12):
            weight *= 1.65
        elif current.month == 1:
            weight *= 0.82
        elif current.month in (7, 8):
            weight *= 1.12
        if current.month == 11 and 20 <= current.day <= 30:
            weight *= 1.35
        days.append(current)
        weights.append(weight)
        current += timedelta(days=1)
    return days, _cumulative_weights(weights)


def _customer_weight(profile: _CustomerProfile, month: int) -> float:
    behavior = profile["behavior"]
    intensity = profile["intensity"]
    if behavior == "declining" and month >= profile["decline_start_month"]:
        months_declining = month - profile["decline_start_month"]
        return intensity * max(0.12, 1.0 - 0.075 * months_declining)
    if behavior == "inactive" and month >= profile["inactive_start_month"]:
        return 0.0
    return intensity


def _make_products(
    rng: random.Random, product_count: int
) -> tuple[list[ProductRecord], dict[str, list[str]], dict[str, Decimal]]:
    categories = tuple(CATEGORIES)
    products: list[ProductRecord] = []
    products_by_category = {category: [] for category in categories}
    retail_prices: dict[str, Decimal] = {}

    for index in range(product_count):
        product_id = f"P{index + 1:04d}"
        category = categories[index % len(categories)]
        minimum_cost, maximum_cost, markup = CATEGORIES[category]
        cost = Decimal(str(round(rng.uniform(minimum_cost, maximum_cost), 2)))
        products.append(
            {
                "product_id": product_id,
                "category": category,
                "brand": rng.choice(BRANDS),
                "cost": cost,
            }
        )
        products_by_category[category].append(product_id)
        retail_prices[product_id] = (cost * Decimal(str(markup))).quantize(
            Decimal("0.01")
        )

    return products, products_by_category, retail_prices


def _make_customers(
    rng: random.Random, customer_count: int, start_date: date
) -> tuple[list[CustomerRecord], list[_CustomerProfile]]:
    signup_start = start_date - timedelta(days=5 * 365)
    signup_days = (start_date - signup_start).days
    categories = tuple(CATEGORIES)
    behavior_choices = ("frequent", "regular", "occasional", "declining", "inactive")
    behavior_weights = (0.18, 0.40, 0.22, 0.12, 0.08)
    base_intensity = {
        "frequent": 4.8,
        "regular": 2.1,
        "occasional": 0.7,
        "declining": 1.8,
        "inactive": 1.8,
    }

    customers: list[CustomerRecord] = []
    profiles: list[_CustomerProfile] = []
    for index in range(customer_count):
        customer_id = f"C{index + 1:06d}"
        behavior = rng.choices(behavior_choices, weights=behavior_weights, k=1)[0]
        customers.append(
            {
                "customer_id": customer_id,
                "age": rng.choices(
                    (18, 24, 35, 50, 65, 75),
                    weights=(0.08, 0.18, 0.30, 0.25, 0.15, 0.04),
                    k=1,
                )[0]
                + rng.randrange(0, 6),
                "gender": rng.choices(
                    ("Female", "Male", "Non-binary", "Prefer not to say"),
                    weights=(0.48, 0.46, 0.03, 0.03),
                    k=1,
                )[0],
                "region": rng.choice(REGIONS),
                "signup_date": signup_start + timedelta(days=rng.randrange(signup_days + 1)),
            }
        )
        profiles.append(
            {
                "behavior": behavior,
                "intensity": base_intensity[behavior] * rng.lognormvariate(0.0, 0.35),
                "favorite_category": rng.choice(categories),
                "decline_start_month": rng.randrange(7, 15),
                "inactive_start_month": rng.randrange(5, 19),
            }
        )
    return customers, profiles


def _make_transactions(
    rng: random.Random,
    transaction_count: int,
    start_date: date,
    end_date: date,
    customers: list[CustomerRecord],
    profiles: list[_CustomerProfile],
    products_by_category: dict[str, list[str]],
    retail_prices: dict[str, Decimal],
    days: list[date],
    day_cumulative: list[float],
) -> list[TransactionRecord]:
    month_count = _month_index(end_date, start_date) + 1
    customer_cumulative_by_month = []
    for month in range(month_count):
        customer_cumulative_by_month.append(
            _cumulative_weights(
                [_customer_weight(profile, month) for profile in profiles]
            )
        )

    categories = tuple(CATEGORIES)
    quantity_choices = (1, 2, 3, 4, 5, 6)
    quantity_weights = (0.61, 0.20, 0.09, 0.045, 0.03, 0.025)
    transactions: list[TransactionRecord] = []
    for index in range(transaction_count):
        transaction_date = days[_weighted_index(rng, day_cumulative)]
        month = _month_index(transaction_date, start_date)
        customer_index = _weighted_index(rng, customer_cumulative_by_month[month])
        profile = profiles[customer_index]

        if rng.random() < 0.7:
            category = profile["favorite_category"]
        else:
            category = rng.choice(categories)
        product_id = rng.choice(products_by_category[category])

        discount_choices = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
        discount_weights = (
            (0.20, 0.15, 0.22, 0.16, 0.13, 0.08, 0.05, 0.01)
            if transaction_date.month in (11, 12)
            else (0.34, 0.20, 0.20, 0.12, 0.08, 0.04, 0.018, 0.002)
        )
        price = (
            retail_prices[product_id]
            * Decimal(str(rng.uniform(0.93, 1.07)))
        ).quantize(Decimal("0.01"))
        transactions.append(
            {
                "transaction_id": f"T{index + 1:08d}",
                "customer_id": customers[customer_index]["customer_id"],
                "transaction_date": transaction_date,
                "product_id": product_id,
                "quantity": rng.choices(
                    quantity_choices, weights=quantity_weights, k=1
                )[0],
                "price": max(price, Decimal("0.01")),
                "discount": Decimal(
                    str(rng.choices(discount_choices, weights=discount_weights, k=1)[0])
                ),
            }
        )
    transactions.sort(key=lambda transaction: transaction["transaction_date"])
    return transactions


def _make_marketing(
    rng: random.Random,
    customers: list[CustomerRecord],
    days: list[date],
    day_cumulative: list[float],
) -> list[MarketingRecord]:
    marketing: list[MarketingRecord] = []
    campaign_weights = (0.18, 0.16, 0.14, 0.12, 0.12, 0.16, 0.12)

    for customer in customers:
        event_count = rng.choices(
            (0, 1, 2, 3, 4, 5),
            weights=(0.20, 0.28, 0.24, 0.15, 0.09, 0.04),
            k=1,
        )[0]
        for _ in range(event_count):
            campaign_date = days[_weighted_index(rng, day_cumulative)]
            marketing.append(
                {
                    "customer_id": customer["customer_id"],
                    "campaign": rng.choices(CAMPAIGNS, weights=campaign_weights, k=1)[0],
                    "campaign_date": campaign_date,
                    "channel": rng.choices(
                        CHANNELS, weights=(0.44, 0.20, 0.20, 0.06, 0.06), k=1
                    )[0],
                }
            )

    marketing.sort(key=lambda event: event["campaign_date"])
    return marketing


def generate_datasets(
    seed: int = DEFAULT_SEED,
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    product_count: int = DEFAULT_PRODUCT_COUNT,
    transaction_count: int = DEFAULT_TRANSACTION_COUNT,
    start_date: date = DEFAULT_START_DATE,
    end_date: date = DEFAULT_END_DATE,
) -> DatasetCollection:
    """Generate related synthetic datasets without writing them to disk."""
    if customer_count < 1:
        raise ValueError("customer_count must be positive")
    if product_count < len(CATEGORIES):
        raise ValueError(f"product_count must be at least {len(CATEGORIES)}")
    if transaction_count < 0:
        raise ValueError("transaction_count cannot be negative")
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")

    rng = random.Random(seed)
    days, day_cumulative = _day_weights(start_date, end_date)
    customers, profiles = _make_customers(rng, customer_count, start_date)
    products, products_by_category, retail_prices = _make_products(rng, product_count)
    transactions = _make_transactions(
        rng,
        transaction_count,
        start_date,
        end_date,
        customers,
        profiles,
        products_by_category,
        retail_prices,
        days,
        day_cumulative,
    )
    marketing = _make_marketing(rng, customers, days, day_cumulative)
    return {
        "customers": customers,
        "products": products,
        "transactions": transactions,
        "marketing": marketing,
    }


def write_datasets(
    datasets: DatasetCollection, output_dir: Path = RAW_DATA_DIR
) -> dict[str, Path]:
    """Write all generated datasets as CSV files and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for name, rows in datasets.items():
        path = output_dir / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=DATASET_FIELDS[name])
            writer.writeheader()
            writer.writerows(rows)
        output_paths[name] = path
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproducible synthetic retail datasets."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--customers", type=int, default=DEFAULT_CUSTOMER_COUNT)
    parser.add_argument("--products", type=int, default=DEFAULT_PRODUCT_COUNT)
    parser.add_argument("--transactions", type=int, default=DEFAULT_TRANSACTION_COUNT)
    parser.add_argument(
        "--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE
    )
    parser.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_END_DATE)
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args()

    datasets = generate_datasets(
        seed=args.seed,
        customer_count=args.customers,
        product_count=args.products,
        transaction_count=args.transactions,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    output_paths = write_datasets(datasets, args.output_dir)
    print(f"Seed: {args.seed}")
    for name, path in output_paths.items():
        print(f"{path.name}: {len(datasets[name])} rows")
    if datasets["transactions"]:
        dates = [record["transaction_date"] for record in datasets["transactions"]]
        print(f"Transaction date range: {min(dates)} to {max(dates)}")


if __name__ == "__main__":
    main()
