#!/usr/bin/env python3
"""Fetch only the four screener sectors + index/metal proxies."""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mf_data_provider import MfDataProvider  # noqa: E402

TOTAL_SUBS = {"Contra Fund", "Flexi Cap Fund", "Focused Fund", "Multi Cap Fund", "Value Fund"}
SILVER_PROXY = "M_ICPVF"
GOLD_PROXY = "M_SBIGL"


def wanted(row) -> bool:
    sub = str(row.get("subsector") or "")
    if "Mid" in sub and "Cap" in sub:
        return True
    if "Small" in sub and "Cap" in sub:
        return True
    if "Multi Asset" in sub:
        return True
    if sub in TOTAL_SUBS and str(row.get("sector") or "") == "Equity":
        return True
    return row.get("mfId") in {SILVER_PROXY, GOLD_PROXY}


def main() -> None:
    p = MfDataProvider()
    all_mf = p.fetch_all_mf_list(force_refresh=False)
    uni = all_mf[all_mf.apply(wanted, axis=1)].copy()
    extra = [x for x in (SILVER_PROXY, GOLD_PROXY) if x not in set(uni["mfId"])]
    ids = list(dict.fromkeys(list(uni["mfId"]) + extra))
    print(f"Fetching {len(ids)} fund charts + {len(p.INDICES)} indices into {p.data_dir}")

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(p.fetch_mf_chart, mf_id, "max", False): mf_id for mf_id in ids}
        for i, fut in enumerate(as_completed(futs), 1):
            mf_id = futs[fut]
            try:
                df = fut.result()
                ok += 1
                if i % 25 == 0:
                    print(f"  funds {i}/{len(ids)} last={mf_id} rows={len(df)}")
            except Exception as exc:
                fail += 1
                print(f"  FAIL {mf_id}: {exc}")

    for name in p.INDICES:
        try:
            df = p.fetch_index_chart(name, force_refresh=False)
            print(f"  index {name}: {len(df)} rows")
        except Exception as exc:
            print(f"  FAIL index {name}: {exc}")
            fail += 1
    print(f"Done ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
