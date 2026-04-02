# AFL Betting Model: Diagnostic Report
**Date:** 2026-03-31 12:50 AWST  
**Status:** Critical data pipeline issues identified

---

## 1. Original Historical Data (`player_game_stats.csv`)

**Expected:** ~273,166 rows (from 2020-2025 season totals expanded to per-game)  
**Actual:** File appears corrupted or incomplete (wc reported 1 line? head returns empty)  
**Impact:** Cannot use as source; need to regenerate from raw season totals (`afltables_all_totals.csv`)

**Action:** Discard current `data/processed/player_game_stats.csv` and rebuild from scratch using the 3,424 season totals we have.

---

## 2. Fixture Scraping Issues

### 2026: ✓ Working (227 matches, all 18 teams, rounds 1-25)
- Source: `data/raw/fixture_2026.csv` and `fixture_mapping_2026.csv` verified
- All team names present, includes byes implicitly

### 2025: ✗ Broken (only 25 matches)
- Likely page structure differs (tables not detected by our generic parser)
- Need to inspect 2025 season page HTML and adjust parser accordingly

### 2020-2024: Not yet attempted
- Should use same parser as 2025 once fixed

**Impact:** Historical enrichment for 2020-2024 cannot proceed without correct season fixture mappings.

---

## 3. Team Name Matching Mismatches

Our season totals use names like:
- "Brisbane" vs "Brisbane Lions"
- "Greater Western Sydney" vs "GWS"
- "Western Bulldogs" vs "Western Bulldogs" (likely same)
- "Gold Coast" vs "Gold Coast Suns" (maybe)
- "North Melbourne" vs "North Melbourne Kangaroos" (maybe)

The fixture mappings include full team names (e.g., "Brisbane Lions"). Our matching in `generate_clean_history.py` tried a simple normalization but may have failed for many teams, causing the 2,724 row count instead of 86k.

---

## 4. Historical Enrichment Failure

`hist_with_context.csv` has only 2,724 rows vs expected ~86k. That means only a tiny fraction of `(year, team)` groups found a match in the fixture mapping.

**Diagnosis steps:**
- Check which `(year, team)` combinations from `afltables_all_totals.csv` are present
- Compare with available `fixture_mapping_<year>.csv` team names
- Count mismatches

---

## 5. 2026 Enrichment Success

`2026_with_context.csv` has 337 rows (15 teams, rounds 1-4). This is lower than expected because:
- Only 15 teams had gb pages (Port Adelaide, Sydney, Western Bulls missing)
- Round 4 may not be completed or some players missing
But the matching logic worked for 2026 because we used the 2026 mapping which is complete.

---

## Root Cause Summary

Primary blockers:
1. Corrupted or missing `player_game_stats.csv` → must regenerate
2. Fixture parser fails for pre-2026 seasons → fix parser or hardcode known team names
3. Team name normalization insufficient → build comprehensive mapping table

---

## Recommended Fix Plan

### Phase 1: Regenerate clean historical per-game dataset
- Re-run `generate_clean_history.py` with improvements:
  - Load `afltables_all_totals.csv` (3,424 rows)
  - For each `(year, team)`, load corresponding `fixture_mapping_<year>.csv` (once fixed)
  - Use robust team name normalization dictionary:
    ```
    "Brisbane Lions" -> "Brisbane"
    "Greater Western Sydney" -> "GWS"
    "Western Bulldogs" -> "Western Bulldogs"
    "Gold Coast Suns" -> "Gold Coast"
    "North Melbourne Kangaroos" -> "North Melbourne"
    "Port Adelaide Power" -> "Port Adelaide"
    "Sydney Swans" -> "Sydney"
    ```
  - Log any `(year, team)` not found and raise error if significant
- Ensure we get ~86k rows

### Phase 2: Fix fixture parsers for 2020-2025
- Inspect `https://afltables.com/afl/seas/2025.html` HTML structure
- Adjust `scrape_fixtures_all.py` to handle tables without obvious round headers or with different class attributes
- Option: Pre-seed round number based on sequence of tables (rounds increase as we go down)
- Validate that each year produces ~165-170 matches (22-23 rounds × ~9 matches per round)

### Phase 3: Validate context features and re-run backtest
- Run `add_all_context.py` on clean historical data
- Verify `hist_with_context.csv` has 86k rows and all context columns non-null for most rows
- Run `backtest_context.py` and compare metrics to baseline

---

## Questions for You

1. Should I proceed with the **fix plan** above (repair historical data pipeline) before adding weather/odds?
2. Do you have a known good mapping of AFL team official names vs common abbreviations? Or should I derive from the 2026 mapping?
3. Is it acceptable to focus solely on 2020-2025 using the season totals expansion (synthetic per-game), recognizing it's an approximation but better than nothing?

---

**Next immediate action I propose:**  
Create a comprehensive team name normalization map and re-run enrichment with logging of missing teams. Then produce a fresh baseline backtest on clean 86k rows to get accurate numbers before adding context.
