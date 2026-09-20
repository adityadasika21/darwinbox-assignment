"""Download real public banking data for an external-validity check.

Every other fixture in this repo is synthetic and therefore only contains the
corruptions I thought of. This pulls genuinely foreign data -- schemas, key names,
date formats and null conventions I did not choose -- from two US government APIs:

    FDIC BankFind   institutions / locations / financials / failures, keyed by CERT
    Treasury Fiscal debt-to-penny and operating cash balance, keyed by record_date

    python eval/fetch_gov.py --out eval/fixtures/gov
    python eval/fetch_gov.py --wide --out eval/fixtures/gov   # + ~36 Socrata sets

Requires network. Nothing else in the repo does, and `make eval` does not depend on
this -- it is an optional, separately-run check.

Note on the CERT range filter: the FDIC endpoints return different default slices,
so an unfiltered pull gives near-disjoint populations across files and any "missed"
relationship is an artefact of the download rather than of the system. Filtering all
endpoints to one CERT range is what an analyst assembling a multi-file extract does.
Key names, date formats and null conventions are left exactly as served.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import urllib.parse
import urllib.request

FDIC = "https://banks.data.fdic.gov/api"
TREASURY = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
CERT_RANGE = "CERT:[10000 TO 12000]"

SOURCES: dict[str, str] = {
    "institutions": f"{FDIC}/institutions?filters={{r}}"
    "&fields=CERT,NAME,CITY,STNAME,ASSET,DEP,BKCLASS,ACTIVE,ESTYMD&limit=900&format=csv",
    "branches": f"{FDIC}/locations?filters={{r}}"
    "&fields=CERT,NAME,CITY,STNAME,SERVTYPE_DESC,ESTYMD&limit=900&format=csv",
    "financials": f"{FDIC}/financials?filters={{r}}"
    "&fields=CERT,REPDTE,ASSET,DEP,NETINC,EQ&limit=900&format=csv",
    "failures": f"{FDIC}/failures"
    "?fields=CERT,NAME,CITYST,FAILDATE,QBFASSET,QBFDEP,RESTYPE&limit=1500&format=csv",
    "treasury_debt": f"{TREASURY}/v2/accounting/od/debt_to_penny"
    "?format=csv&page[size]=400&sort=-record_date",
    "treasury_cash": f"{TREASURY}/v1/accounting/dts/operating_cash_balance"
    "?format=csv&page[size]=400&sort=-record_date",
}

# What the system should find. Keys are (left, right) alias.column, order-insensitive.
EXPECTED = [
    ("financials.cert", "institutions.cert"),
    ("branches.cert", "institutions.cert"),
    ("treasury_cash.record_date+record_fiscal_quarter",
     "treasury_debt.record_date+record_fiscal_quarter"),
]


# The wide corpus. The banking files above have known relationships and act as
# signal; every dataset below is from an unrelated domain, so any edge discovered
# between two of them is a false positive by construction. That is the whole point --
# precision cannot be measured on data assembled to relate.
#
# Datasets are discovered through the Socrata catalog rather than pinned by id,
# because ids rot: a pinned list degrades silently into "3 of 36 downloaded". The
# search terms are fixed, so the *shape* of the corpus is stable even as the
# particular datasets behind it change.
SOCRATA_CATALOG = "https://api.us.socrata.com/api/catalog/v1"
SOCRATA_TOPICS = [
    "restaurant inspections", "traffic crashes", "school enrollment", "air quality",
    "farmers markets", "election results", "employee salaries", "energy consumption",
    "water quality", "bus ridership", "building permits", "library circulation",
    "park facilities", "fire incidents", "housing units", "business licenses",
    "vehicle registrations", "hospital discharges",
]
SOCRATA_ROWS = 800


def socrata_datasets(topics: list[str], per_topic: int) -> list[tuple[str, str]]:
    """Return (name, csv_url) for public tabular datasets matching each topic."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for topic in topics:
        query = urllib.parse.urlencode(
            {"q": topic, "only": "dataset", "limit": per_topic * 3}
        )
        try:
            payload = json.loads(fetch(f"{SOCRATA_CATALOG}?{query}"))
        except Exception as exc:  # noqa: BLE001 - one dead topic must not stop the rest
            print(f"  catalog {topic:<22} FAILED: {exc}")
            continue

        taken = 0
        for result in payload.get("results", []):
            resource = result.get("resource", {})
            domain = result.get("metadata", {}).get("domain")
            dataset_id = resource.get("id")
            # Needs real columns to be worth anything as a false-positive source.
            if not domain or not dataset_id or len(resource.get("columns_name") or []) < 4:
                continue
            if dataset_id in seen:
                continue
            seen.add(dataset_id)
            slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
            name = f"{slug}_{taken}" if taken else slug
            found.append(
                (name, f"https://{domain}/resource/{dataset_id}.csv?$limit={SOCRATA_ROWS}")
            )
            taken += 1
            if taken >= per_topic:
                break
    return found


def fetch(url: str) -> bytes:
    encoded = urllib.parse.quote(url, safe=":/?&=[]%,")
    request = urllib.request.Request(encoded, headers={"User-Agent": "darwinbox-eval/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("eval/fixtures/gov"))
    parser.add_argument(
        "--wide",
        action="store_true",
        help="also pull ~36 unrelated Socrata datasets, for the precision check",
    )
    parser.add_argument("--per-topic", type=int, default=2)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-query the catalog instead of reusing manifest.json",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, template in SOURCES.items():
        url = template.replace("{r}", urllib.parse.quote(CERT_RANGE))
        try:
            data = fetch(url)
        except Exception as exc:  # noqa: BLE001 - a dead endpoint must name itself
            print(f"  {name:<16} FAILED: {exc}")
            continue
        path = args.out / f"{name}.csv"
        path.write_bytes(data)
        print(f"  {name:<16} {len(data.splitlines()):>5} lines -> {path}")

    if args.wide:
        # The catalog is queried once and the result written to a manifest, so a later
        # run rebuilds the *same* corpus rather than whatever the catalog ranks highest
        # that week. Without it the measured precision is reproducible but not
        # repeatable, and a number that moves on its own is hard to act on.
        manifest = args.out / "manifest.json"
        if manifest.exists() and not args.refresh:
            chosen = [tuple(row) for row in json.loads(manifest.read_text())]
            print(f"\nSocrata (from {manifest}; --refresh to re-discover):")
        else:
            chosen = socrata_datasets(SOCRATA_TOPICS, args.per_topic)
            manifest.write_text(json.dumps(chosen, indent=2))
            print("\nSocrata (unrelated domains -- any edge between these is a false positive):")

        for name, url in chosen:
            try:
                data = fetch(url)
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:<24} FAILED: {exc}")
                continue
            if len(data.splitlines()) < 3:
                print(f"  {name:<24} skipped (no rows)")
                continue
            path = args.out / f"{name}.csv"
            path.write_bytes(data)
            print(f"  {name:<24} {len(data.splitlines()):>5} lines -> {path}")

    print("\nExpected relationships:")
    for left, right in EXPECTED:
        print(f"  {left} = {right}")
    print("\nCheck with:  python -m darwinbox.cli eval/fixtures/gov/*.csv --schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
