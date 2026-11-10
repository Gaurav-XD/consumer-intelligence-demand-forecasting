"""Profile generated CSV datasets and report data-quality issues."""

import csv
import json
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, TypedDict, get_type_hints

from consumer_intelligence.config import RAW_DATA_DIR, REPORTS_DIR
from consumer_intelligence.data_generator import (
    BRANDS,
    CAMPAIGNS,
    CHANNELS,
    CATEGORIES,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    REGIONS,
)
from consumer_intelligence.data_schema import (
    CustomerRecord,
    MarketingRecord,
    ProductRecord,
    TransactionRecord,
)

REPORT_PATH = REPORTS_DIR / "data_quality_report.json"
EXPECTED_SCHEMAS = {
    "customers": get_type_hints(CustomerRecord),
    "products": get_type_hints(ProductRecord),
    "transactions": get_type_hints(TransactionRecord),
    "marketing": get_type_hints(MarketingRecord),
}
PRIMARY_KEYS = {
    "customers": ("customer_id",),
    "products": ("product_id",),
    "transactions": ("transaction_id",),
    "marketing": (),
}
DATE_FIELDS = {
    "customers": ("signup_date",),
    "transactions": ("transaction_date",),
    "marketing": ("campaign_date",),
    "products": (),
}
CATEGORICAL_VALUES = {
    ("customers", "gender"): {"Female", "Male", "Non-binary", "Prefer not to say"},
    ("customers", "region"): set(REGIONS),
    ("products", "category"): set(CATEGORIES),
    ("products", "brand"): set(BRANDS),
    ("marketing", "campaign"): set(CAMPAIGNS),
    ("marketing", "channel"): set(CHANNELS),
}
_TYPE_NAMES = {
    str: "string",
    int: "integer",
    Decimal: "decimal",
    date: "date",
}


class CsvDataset(TypedDict):
    fields: list[str]
    rows: list[dict[str, str | None]]


def _read_dataset(path: Path) -> CsvDataset:
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return {"fields": [], "rows": []}
        return {"fields": reader.fieldnames, "rows": list(reader)}


def _is_missing(value: str | None) -> bool:
    return value is None or not value.strip()


def _parse_date(value: str | None) -> date | None:
    if _is_missing(value):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _parse_decimal(value: str | None) -> Decimal | None:
    if _is_missing(value):
        return None
    try:
        parsed = Decimal(value.strip())
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _parse_integer(value: str | None) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def _observed_type(value: str) -> str:
    if _parse_date(value) is not None:
        return "date"
    if _parse_integer(value) is not None:
        return "integer"
    if _parse_decimal(value) is not None:
        return "decimal"
    return "string"


def _add_issue(
    issues: list[dict[str, int | str]], check: str, count: int
) -> None:
    if count:
        issues.append({"check": check, "count": count})


def _dataset_report(
    name: str, dataset: CsvDataset, all_datasets: dict[str, CsvDataset]
) -> dict[str, Any]:
    fields = dataset["fields"]
    rows = dataset["rows"]
    schema = EXPECTED_SCHEMAS[name]
    issues: list[dict[str, int | str]] = []

    missing_columns = set(schema) - set(fields)
    unexpected_columns = set(fields) - set(schema)
    if missing_columns or unexpected_columns or tuple(fields) != tuple(schema):
        mismatch_count = len(missing_columns) + len(unexpected_columns)
        if not mismatch_count:
            mismatch_count = 1
        _add_issue(issues, "column_schema_mismatch", mismatch_count)

    missing_values = {
        field: sum(_is_missing(row.get(field)) for row in rows) for field in fields
    }
    for field, missing_count in missing_values.items():
        _add_issue(issues, f"missing_{field}", missing_count)
    duplicate_rows = 0
    seen_rows: set[tuple[str | None, ...]] = set()
    for row in rows:
        row_values = tuple(row.get(field) for field in fields)
        if row_values in seen_rows:
            duplicate_rows += 1
        seen_rows.add(row_values)
    _add_issue(issues, "duplicate_rows", duplicate_rows)

    data_types: dict[str, dict[str, str]] = {}
    for field, expected_type in schema.items():
        values = [row.get(field) for row in rows if not _is_missing(row.get(field))]
        observed_types = {_observed_type(value) for value in values if value is not None}
        observed_type = (
            next(iter(observed_types))
            if len(observed_types) == 1
            else "mixed" if observed_types else "unknown"
        )
        data_types[field] = {
            "expected": _TYPE_NAMES[expected_type],
            "observed": observed_type,
        }

    unique_key_counts: dict[str, int] = {}
    for field in fields:
        if field.endswith("_id"):
            unique_key_counts[field] = len(
                {
                    row.get(field)
                    for row in rows
                    if not _is_missing(row.get(field))
                }
            )
    for key in PRIMARY_KEYS[name]:
        if key in fields:
            key_counts = Counter(
                row.get(key) for row in rows if not _is_missing(row.get(key))
            )
            duplicate_keys = sum(count - 1 for count in key_counts.values() if count > 1)
            _add_issue(issues, f"duplicate_{key}", duplicate_keys)
            _add_issue(
                issues,
                f"missing_{key}",
                sum(_is_missing(row.get(key)) for row in rows),
            )

    date_ranges: dict[str, dict[str, str | None]] = {}
    invalid_dates: dict[str, int] = {}
    for field in DATE_FIELDS[name]:
        parsed_dates = [
            parsed
            for row in rows
            if (parsed := _parse_date(row.get(field))) is not None
        ]
        invalid_count = sum(
            not _is_missing(row.get(field))
            and _parse_date(row.get(field)) is None
            for row in rows
        )
        invalid_dates[field] = invalid_count
        date_ranges[field] = {
            "minimum": min(parsed_dates).isoformat() if parsed_dates else None,
            "maximum": max(parsed_dates).isoformat() if parsed_dates else None,
        }
        _add_issue(issues, f"invalid_{field}", invalid_count)

    invalid_numeric_values: dict[str, int] = {}

    def numeric_violations(
        field: str, invalid_predicate: Callable[[int | Decimal], bool]
    ) -> int:
        if field not in fields:
            return 0
        invalid_count = 0
        for row in rows:
            value = row.get(field)
            if _is_missing(value):
                continue
            parsed = (
                _parse_integer(value)
                if schema[field] is int
                else _parse_decimal(value)
            )
            if parsed is None or invalid_predicate(parsed):
                invalid_count += 1
        return invalid_count

    numeric_rules: dict[str, Callable[[int | Decimal], bool]] = {}
    if name == "customers":
        numeric_rules["age"] = lambda value: value < 13 or value > 100
    elif name == "products":
        numeric_rules["cost"] = lambda value: value <= 0
    elif name == "transactions":
        numeric_rules.update(
            {
                "quantity": lambda value: value <= 0,
                "price": lambda value: value <= 0,
                "discount": lambda value: value < 0 or value > 1,
            }
        )
    for field, invalid_predicate in numeric_rules.items():
        count = numeric_violations(field, invalid_predicate)
        invalid_numeric_values[field] = count
        _add_issue(issues, f"invalid_{field}", count)

    invalid_categorical_values: dict[str, int] = {}
    for (dataset_name, field), allowed_values in CATEGORICAL_VALUES.items():
        if dataset_name != name or field not in fields:
            continue
        invalid_count = sum(
            not _is_missing(row.get(field)) and row.get(field) not in allowed_values
            for row in rows
        )
        invalid_categorical_values[field] = invalid_count
        _add_issue(issues, f"invalid_{field}", invalid_count)

    referential_integrity: dict[str, int] = {}
    reference_specs = {
        "transactions": {
            "customer_id": ("customers", "customer_id"),
            "product_id": ("products", "product_id"),
        },
        "marketing": {"customer_id": ("customers", "customer_id")},
    }
    for field, (parent_name, parent_field) in reference_specs.get(name, {}).items():
        referential_integrity[field] = 0
        if field not in fields or parent_field not in EXPECTED_SCHEMAS[parent_name]:
            continue
        parent_rows = all_datasets[parent_name]["rows"]
        valid_ids = {
            row.get(parent_field)
            for row in parent_rows
            if not _is_missing(row.get(parent_field))
        }
        invalid_count = sum(
            _is_missing(row.get(field)) or row.get(field) not in valid_ids
            for row in rows
        )
        referential_integrity[field] = invalid_count
        _add_issue(issues, f"invalid_{field}_reference", invalid_count)

    if name == "transactions" and "transaction_date" in fields:
        outside_period = 0
        for row in rows:
            parsed = _parse_date(row.get("transaction_date"))
            if parsed is not None and not DEFAULT_START_DATE <= parsed <= DEFAULT_END_DATE:
                outside_period += 1
        _add_issue(issues, "transaction_date_outside_expected_period", outside_period)

    return {
        "row_count": len(rows),
        "column_count": len(fields),
        "columns": fields,
        "data_types": data_types,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "unique_key_counts": unique_key_counts,
        "date_ranges": date_ranges,
        "invalid_dates": invalid_dates,
        "invalid_numeric_values": invalid_numeric_values,
        "invalid_categorical_values": invalid_categorical_values,
        "referential_integrity": referential_integrity,
        "issues": issues,
        "status": "PASS" if not issues else "FAIL",
    }


def run_data_quality_report(
    data_dir: Path = RAW_DATA_DIR, report_path: Path = REPORT_PATH
) -> dict[str, Any]:
    """Validate all four CSV datasets and save a JSON quality report."""
    data_dir = Path(data_dir)
    report_path = Path(report_path)
    expected_files = {
        "customers": "customers.csv",
        "products": "products.csv",
        "transactions": "transactions.csv",
        "marketing": "marketing.csv",
    }
    datasets = {
        name: _read_dataset(data_dir / filename)
        for name, filename in expected_files.items()
    }

    reports = {
        name: _dataset_report(name, dataset, datasets)
        for name, dataset in datasets.items()
    }
    issue_count = sum(
        issue["count"]
        for dataset_report in reports.values()
        for issue in dataset_report["issues"]
    )
    report: dict[str, Any] = {
        "overall_status": "PASS" if issue_count == 0 else "FAIL",
        "issue_count": issue_count,
        "expected_transaction_period": {
            "start": DEFAULT_START_DATE.isoformat(),
            "end": DEFAULT_END_DATE.isoformat(),
        },
        "datasets": reports,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)
        json_file.write("\n")
    return report


if __name__ == "__main__":
    quality_report = run_data_quality_report()
    print(
        f"Data quality: {quality_report['overall_status']} "
        f"({quality_report['issue_count']} issues)"
    )
    print(f"Report saved to: {REPORT_PATH}")
