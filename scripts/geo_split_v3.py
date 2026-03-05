import pandas as pd
import numpy as np
import zipfile
import io
import os

# ─── PATHS ────────────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
BRAINTREE_CSV = os.path.join(BASE_DIR, 'Braintree data - March 2026 - Sheet1.csv')
ZIP_PATH      = os.path.join(BASE_DIR, '023882f8d77741f4d5347f80d95bc259-f9f3424dbe4fb58b3dac65dced4c1c3a0f0db27a.zip')
ZIP_ENTRY     = '023882f8d77741f4d5347f80d95bc259-f9f3424dbe4fb58b3dac65dced4c1c3a0f0db27a/Zip Codes to DMAs'

# ─── 1. LOAD & PREPARE BRAINTREE DATA ─────────────────────────────────────────

print("=" * 70)
print("GEO SPLIT v3 — BRAINTREE DATA PREP")
print("=" * 70)

print("\n[1] Loading Braintree CSV...")
raw = pd.read_csv(BRAINTREE_CSV, dtype=str)
raw.columns = raw.columns.str.strip()
print(f"    Total rows loaded: {len(raw):,}")

# Filter United States only
us = raw[raw['Billing Country'].str.strip() == 'United States of America'].copy()
print(f"    US rows: {len(us):,}  ({len(us)/len(raw)*100:.1f}% of total)")

# Exclude credit transactions
credits = us['Transaction Type'].str.strip().str.lower() == 'credit'
print(f"    Credit rows excluded: {credits.sum():,}")
us = us[~credits].copy()
print(f"    Rows after credit filter: {len(us):,}")

# Parse date and derive week (Monday-anchored)
us['date'] = pd.to_datetime(us['Created Datetime'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
us['week'] = us['date'] - pd.to_timedelta(us['date'].dt.dayofweek, unit='d')
us['week'] = us['week'].dt.normalize()   # strip time component

# Clean numeric columns
us['Amount Authorized'] = pd.to_numeric(us['Amount Authorized'], errors='coerce')

# Clean postal code: keep first 5 digits, zero-pad if short
us['postal_code'] = (
    us['Billing Postal Code']
    .str.strip()
    .str.split('-').str[0]          # drop ZIP+4 suffix
    .str.zfill(5)                   # zero-pad to 5 chars
)

print(f"    Date range: {us['date'].min().date()} → {us['date'].max().date()}")
print(f"    Unique postal codes: {us['postal_code'].nunique():,}")

# ─── 2. WEEKLY POSTAL CODE AGGREGATION ────────────────────────────────────────

print("\n[2] Aggregating to weekly postal-code level...")

weekly = (
    us.groupby(['week', 'postal_code'], sort=True)
    .agg(
        bookings=('Transaction ID', 'nunique'),   # distinct transaction IDs
        sales   =('Amount Authorized', 'sum'),
    )
    .reset_index()
)

print(f"    Unique weeks: {weekly['week'].nunique()}")
print(f"    Unique postal codes: {weekly['postal_code'].nunique():,}")
print(f"    Weekly postal-code rows: {len(weekly):,}")
print(weekly.head(10).to_string(index=False))

# ─── 3. LOAD ZIP → DMA MAPPING ────────────────────────────────────────────────

print("\n[3] Loading ZIP → DMA mapping from zip archive...")

with zipfile.ZipFile(ZIP_PATH) as zf:
    with zf.open(ZIP_ENTRY) as f:
        dma_map = pd.read_csv(f, sep='\t', dtype=str)

dma_map.columns = dma_map.columns.str.strip()
dma_map['zip_code'] = dma_map['zip_code'].str.strip().str.zfill(5)
dma_map = dma_map.rename(columns={'zip_code': 'postal_code'})

print(f"    Mapping rows loaded: {len(dma_map):,}")
print(f"    Unique ZIP codes in mapping: {dma_map['postal_code'].nunique():,}")

# ─── 4. MERGE DMA INTO WEEKLY DATA ────────────────────────────────────────────

print("\n[4] Mapping postal codes to DMA...")

weekly = weekly.merge(dma_map, on='postal_code', how='left')

mapped   = weekly['dma_code'].notna()
unmapped = ~mapped

print(f"    Rows mapped to DMA:   {mapped.sum():,}")
print(f"    Rows unmapped:        {unmapped.sum():,}")

# ─── 5. HIGHLIGHT UNMAPPED POSTAL CODES ───────────────────────────────────────

unmapped_codes = (
    weekly.loc[unmapped, ['postal_code', 'bookings', 'sales']]
    .groupby('postal_code', sort=False)
    .agg(total_bookings=('bookings', 'sum'), total_sales=('sales', 'sum'))
    .sort_values('total_bookings', ascending=False)
    .reset_index()
)

print(f"\n[5] Unmapped postal codes: {len(unmapped_codes)}")
if len(unmapped_codes):
    print(unmapped_codes.to_string(index=False))
else:
    print("    All postal codes mapped successfully.")

# Flag in main dataframe for easy filtering
weekly['dma_mapped'] = mapped

# ─── OUTPUT ───────────────────────────────────────────────────────────────────

print("\n[OUTPUT] Final weekly dataframe columns:", weekly.columns.tolist())
print(weekly.head(10).to_string(index=False))

OUT_PATH = os.path.join(BASE_DIR, 'geo_split_v3_weekly.csv')
weekly.to_csv(OUT_PATH, index=False)
print(f"\n[SAVED] {OUT_PATH}")
print("Done.")
