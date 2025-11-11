"""
Export 10 datasets to JSON format from backups
Simple version using backed up Parquet files
"""
import json
import pandas as pd
from pathlib import Path

def has_baltimore_data(df: pd.DataFrame) -> dict:
    """Check if dataset contains Baltimore/Maryland data"""
    cols_lower = [col.lower() for col in df.columns]

    # Check for state columns
    state_cols = ['state', 'stateabbr', 'state_abbr', 'statedesc', 'state_desc', 'locationabbr', 'locationdesc']
    has_state = any(col in cols_lower for col in state_cols)

    # Check for location/county columns
    location_cols = ['county', 'location', 'countyname', 'locationdesc', 'fips', 'countyfips']
    has_location = any(col in cols_lower for col in location_cols)

    # Try to count Maryland data
    md_count = 0
    baltimore_count = 0

    if has_state:
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in state_cols:
                try:
                    md_count = int(df[col].astype(str).str.contains('MD|Maryland', case=False, na=False, regex=True).sum())
                    if md_count > 0:
                        break
                except:
                    pass

    if has_location:
        for col in df.columns:
            col_lower = col.lower()
            if any(loc in col_lower for loc in location_cols):
                try:
                    baltimore_count = int(df[col].astype(str).str.contains('Baltimore', case=False, na=False).sum())
                    if baltimore_count > 0:
                        break
                except:
                    pass

    return {
        "has_state_col": has_state,
        "has_location_col": has_location,
        "maryland_rows": md_count,
        "baltimore_rows": baltimore_count,
        "is_baltimore_dataset": baltimore_count > 0 or md_count > 0
    }

def export_parquet_to_json(parquet_path: Path, output_dir: Path):
    """Export a Parquet file to JSON"""
    try:
        # Read parquet
        df = pd.read_parquet(parquet_path)

        # Get dataset info from path
        parts = parquet_path.parts
        org = parts[-2] if len(parts) >= 2 else "Unknown"
        uid = parquet_path.stem

        print(f"\n[EXPORT] {org}/{uid}")
        print(f"         Rows: {len(df):,} | Columns: {len(df.columns)}")

        # Check for Baltimore data
        baltimore_info = has_baltimore_data(df)

        if baltimore_info["is_baltimore_dataset"]:
            print(f"         [BALTIMORE DATA FOUND!]")
            print(f"         Maryland rows: {baltimore_info['maryland_rows']:,}")
            print(f"         Baltimore rows: {baltimore_info['baltimore_rows']:,}")

        # Create filename
        json_filename = f"{org}_{uid}.json"
        json_path = output_dir / json_filename

        # Convert to JSON
        data_dict = {
            "metadata": {
                "dataset_name": f"{org}:{uid}",
                "uid": uid,
                "organization": org,
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "baltimore_info": baltimore_info,
                "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
            },
            "data": json.loads(df.to_json(orient="records", date_format="iso"))
        }

        # Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)

        file_size = json_path.stat().st_size / (1024 * 1024)
        print(f"         [SUCCESS] Exported: {json_filename} ({file_size:.2f} MB)")

        return {
            "success": True,
            "org": org,
            "uid": uid,
            "filename": json_filename,
            "rows": len(df),
            "baltimore_info": baltimore_info
        }

    except Exception as e:
        print(f"         [ERROR] Failed: {e}")
        return {
            "success": False,
            "path": str(parquet_path),
            "error": str(e)
        }

def main():
    print("="*80)
    print("EXPORTING 10 DATASETS TO JSON FORMAT")
    print("="*80)

    # Find backed up parquet files
    backup_dir = Path("professor-complete-package/backups/minio-data/mh-raw/raw")

    if not backup_dir.exists():
        print(f"\n[ERROR] Backup directory not found: {backup_dir}")
        print("[INFO] Looking for alternative backup location...")
        backup_dir = Path("backups/minio-data/mh-raw/raw")

        if not backup_dir.exists():
            print(f"[ERROR] Also not found: {backup_dir}")
            print("[ERROR] Please run backup_data.py first!")
            return

    print(f"\n[INFO] Reading from: {backup_dir}/")

    # Find all parquet files
    parquet_files = list(backup_dir.rglob("*.parquet"))
    print(f"[INFO] Found {len(parquet_files)} parquet files")

    if len(parquet_files) == 0:
        print("[ERROR] No parquet files found!")
        return

    # Create output directory
    output_dir = Path("team-datasets-json")
    output_dir.mkdir(exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}/\n")

    # Prioritize files that might have Baltimore data
    # BRFSS (Behavioral Risk Factor Surveillance System) typically has state/county data
    baltimore_likely = []
    other_files = []

    for pf in parquet_files:
        if "BRFSS" in str(pf).upper() or "STATE" in str(pf).upper():
            baltimore_likely.append(pf)
        else:
            other_files.append(pf)

    # Combine: Baltimore-likely first, then others
    sorted_files = baltimore_likely + other_files

    # Select top 10
    selected_files = sorted_files[:10]

    print(f"[INFO] Selected 10 datasets for export")
    print(f"[INFO] {len(baltimore_likely)} likely to contain Baltimore data")
    print("\n" + "="*80)
    print("STARTING EXPORT")
    print("="*80)

    # Export each file
    results = []
    baltimore_count = 0

    for pf in selected_files:
        result = export_parquet_to_json(pf, output_dir)
        results.append(result)

        if result.get("success") and result.get("baltimore_info", {}).get("is_baltimore_dataset"):
            baltimore_count += 1

    # Summary
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)

    successful = sum(1 for r in results if r.get("success"))
    print(f"\n[SUCCESS] Exported {successful} out of {len(selected_files)} datasets")
    print(f"[INFO] Datasets with Baltimore/Maryland data: {baltimore_count}")
    print(f"[INFO] Output directory: {output_dir}/")

    # Calculate total size
    json_files = list(output_dir.glob("*.json"))
    total_size = sum(f.stat().st_size for f in json_files) / (1024 * 1024)

    print(f"\n[SUMMARY]")
    print(f"  Files exported: {len(json_files)}")
    print(f"  Total size: {total_size:.2f} MB")
    print(f"  Baltimore datasets: {baltimore_count}")

    # Create README
    readme = f"""# Team Datasets - JSON Format

## Overview

This folder contains 10 complete datasets exported from the Mental Health Platform.

**Total datasets:** {successful}
**Datasets with Baltimore/Maryland data:** {baltimore_count}
**Total size:** {total_size:.2f} MB

---

## Datasets Included:

"""

    for i, result in enumerate(results, 1):
        if result.get("success"):
            readme += f"\n### {i}. {result['org']} - {result['uid']}\n"
            readme += f"- **File:** `{result['filename']}`\n"
            readme += f"- **Rows:** {result['rows']:,}\n"

            balt_info = result.get('baltimore_info', {})
            if balt_info.get('is_baltimore_dataset'):
                readme += f"- **Maryland rows:** {balt_info.get('maryland_rows', 0):,}\n"
                readme += f"- **Baltimore rows:** {balt_info.get('baltimore_rows', 0):,}\n"
                readme += f"- **Contains Baltimore data:** YES\n"
            else:
                readme += f"- **Contains Baltimore data:** No\n"
            readme += "\n"

    readme += """
---

## JSON File Structure:

```json
{
  "metadata": {
    "dataset_name": "CDC:dataset-id",
    "uid": "dataset-id",
    "organization": "CDC/BRFSS/SAMHSA",
    "total_rows": 1000,
    "total_columns": 20,
    "columns": ["column1", "column2", ...],
    "column_types": {"column1": "object", "column2": "int64", ...},
    "baltimore_info": {
      "has_state_col": true,
      "has_location_col": true,
      "maryland_rows": 500,
      "baltimore_rows": 100,
      "is_baltimore_dataset": true
    }
  },
  "data": [
    {"column1": "value1", "column2": "value2", ...},
    ...
  ]
}
```

---

## How to Use:

### Python:
```python
import json
import pandas as pd

# Load a dataset
with open('team-datasets-json/dataset_file.json', 'r') as f:
    dataset = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(dataset['data'])

# Filter for Baltimore data (if applicable)
metadata = dataset['metadata']
if metadata['baltimore_info']['is_baltimore_dataset']:
    # Find location column
    for col in df.columns:
        if 'location' in col.lower() or 'state' in col.lower():
            baltimore_df = df[df[col].str.contains('Baltimore|MD', case=False, na=False)]
            print(f"Baltimore rows: {len(baltimore_df)}")
            break
```

### JavaScript:
```javascript
const fs = require('fs');
const dataset = JSON.parse(fs.readFileSync('dataset_file.json', 'utf8'));

console.log(`Dataset: ${dataset.metadata.dataset_name}`);
console.log(`Rows: ${dataset.metadata.total_rows}`);
console.log('Sample data:', dataset.data.slice(0, 5));
```

---

## Filtering Baltimore/Maryland Data:

For datasets with Baltimore data, look for these columns:
- **State:** `State`, `StateAbbr`, `LocationDesc`, `StateDesc`
- **County:** `County`, `CountyName`, `LocationDesc`
- **FIPS:** Baltimore City = `24510`, Baltimore County = `24005`, Maryland = `24`

Example filters:
```python
# Maryland state data
md_data = df[df['StateAbbr'] == 'MD']

# Baltimore specifically
baltimore = df[df['LocationDesc'].str.contains('Baltimore', case=False, na=False)]
```

---

**Ready to share with your team!**
"""

    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"\n[SUCCESS] Created: {output_dir}/README.md")
    print(f"\n[SUCCESS] Ready to share with team!")
    print("="*80)

if __name__ == "__main__":
    main()
