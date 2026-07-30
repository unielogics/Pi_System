"""
Local product profiles for pass/fail QA validation against a known expected size. Same
local-JSON-file pattern as calibration/device config -- intentionally local to the Pi for now,
not wired into the WMS backend's Item records (that's a deferred follow-up).
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

PRODUCTS_PATH = os.path.join(os.path.dirname(__file__), "data", "products.json")


@dataclass
class Product:
    id: str
    name: str
    expected_length_in: float
    expected_width_in: float
    expected_height_in: float
    tolerance_in: float = 0.5


def list_products() -> list[Product]:
    if not os.path.exists(PRODUCTS_PATH):
        return []
    with open(PRODUCTS_PATH, "r") as f:
        data = json.load(f)
    return [Product(**item) for item in data]


def _save_all(products: list[Product]) -> None:
    os.makedirs(os.path.dirname(PRODUCTS_PATH), exist_ok=True)
    with open(PRODUCTS_PATH, "w") as f:
        json.dump([asdict(p) for p in products], f)


def create_product(
    name: str,
    expected_length_in: float,
    expected_width_in: float,
    expected_height_in: float,
    tolerance_in: float = 0.5,
) -> Product:
    products = list_products()
    product = Product(
        id=uuid.uuid4().hex[:12],
        name=name,
        expected_length_in=expected_length_in,
        expected_width_in=expected_width_in,
        expected_height_in=expected_height_in,
        tolerance_in=tolerance_in,
    )
    products.append(product)
    _save_all(products)
    return product


def get_product(product_id: str) -> Optional[Product]:
    for p in list_products():
        if p.id == product_id:
            return p
    return None


def delete_product(product_id: str) -> bool:
    products = list_products()
    remaining = [p for p in products if p.id != product_id]
    deleted = len(remaining) < len(products)
    if deleted:
        _save_all(remaining)
    return deleted
