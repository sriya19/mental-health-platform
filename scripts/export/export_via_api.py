"""
Export 10 datasets to JSON using backend API
"""
import requests
import json
import pandas as pd
from pathlib import Path
import io

BACKEND_URL = "http://localhost:8000"

def get_datasets():
    """Get list of all datasets"""
    response = requests.get(f"{BACKEND_URL}/datasets?org=All")
    data = response.json()
    return data.get("items", [])

def download_dataset(org: str, uid: str) -> pd.DataFrame:
    """Download dataset as JSON and convert to DataFrame"""
    # Download as JSON
    response = requests.get(f"{BACKEND_URL}/download/{org}/{uid}")

    if response.status_code == 200:
        # Parse JSON
        data = response.json()
        df = pd.DataFrame(data['data'])
        return df
    else:
        raise Exception(f"Download failed: {response.status_code}")

def has_baltimore_data(df: pd.DataFrame) -> dict:
    """Check if dataset contains Baltimore/Maryland data"""
    cols_lower = [col.lower() for col in df.columns]

    state_cols = ['state', 'stateabbr', 'state_abbr', 'statedesc', 'state_desc', 'locationabbr', 'locationdesc']
    has_state = any(col in cols_lower for col in state_cols)

    location_cols = ['county', 'location', 'countyname', 'locationdesc', 'fips', 'countyfips']
    has_location = any(col in cols_lower for col in location_cols)

    md_count = 0
    baltimore_count = 0

    if has_state:
        for col in df.columns:
            if col.lower() in state_cols:
                try:
                    md_count = int(df[col].astype(str).str.contains('MD|Maryland', case=False, na=False, regex=True).sum())
                    if md_count > 0:
                        break
                except:
                    pass

    if has_location:
        for col in df.columns:
            if any(loc in col.lower() for loc in location_cols):
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

def export_dataset(dataset_info: dict, output_dir: Path):
    """Export a single dataset to JSON"""
    uid = dataset_info.get("uid", "")
    name = dataset_info.get("name", "Unknown")
    org = dataset_info.get("org", dataset_info.get("organization", "Unknown"))

    if not uid:
        print(f"\n[SKIP] {name} - No UID")
        return {"success": False, "error": "no_uid"}

    try:
        print(f"\n[EXPORT] {name}")
        print(f"         UID: {uid}")

        # Download dataset
        df = download_dataset(org, uid)
        print(f"         Rows: {len(df):,} | Columns: {len(df.columns)}")

        # Check for Baltimore data
        baltimore_info = has_baltimore_data(df)

        if baltimore_info["is_baltimore_dataset"]:
            print(f"         [BALTIMORE DATA FOUND!]")
            print(f"         Maryland rows: {baltimore_info['maryland_rows']:,}")
            print(f"         Baltimore rows: {baltimore_info['baltimore_rows']:,}")

        # Create safe filename
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
        safe_name = safe_name[:50]
        json_filename = f"{safe_name}_{uid}.json"
        json_path = output_dir / json_filename

        # Create JSON structure
        data_dict = {
            "metadata": {
                "dataset_name": name,
                "uid": uid,
                "organization": org,
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "baltimore_info": baltimore_info
            },
            "data": json.loads(df.to_json(orient="records", date_format="iso"))
        }

        # Write JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)

        file_size = json_path.stat().st_size / (1024 * 1024)
        print(f"         [SUCCESS] Exported: {json_filename} ({file_size:.2f} MB)")

        return {
            "success": True,
            "dataset_name": name,
            "uid": uid,
            "org": org,
            "filename": json_filename,
            "rows": len(df),
            "baltimore_info": baltimore_info
        }

    except Exception as e:
        print(f"         [ERROR] Failed: {e}")
        return {
            "success": False,
            "dataset_name": name,
            "uid": uid,
            "error": str(e)
        }

def main():
    print("="*80)
    print("EXPORTING 10 DATASETS TO JSON FORMAT")
    print("="*80)

    # Create output directory
    output_dir = Path("team-datasets-json")
    output_dir.mkdir(exist_ok=True)
    print(f"\n[INFO] Output directory: {output_dir}/\n")

    # Get all datasets
    print("[INFO] Fetching datasets from backend...")
    datasets = get_datasets()
    print(f"[INFO] Found {len(datasets)} total datasets")

    # Prioritize datasets likely to have location data
    location_keywords = ['state', 'county', 'location', 'geographic', 'brfss', 'behavioral', 'surveillance']

    def score_dataset(ds):
        """Score dataset by likelihood of having Baltimore data"""
        score = 0
        name = ds.get('name', '').lower()
        desc = ds.get('description', '').lower() if ds.get('description') else ''

        for keyword in location_keywords:
            if keyword in name or keyword in desc:
                score += 1

        return score

    # Sort by score
    datasets_sorted = sorted(datasets, key=score_dataset, reverse=True)

    # Select top 10
    selected_datasets = datasets_sorted[:10]

    print(f"[INFO] Selected 10 datasets for export:\n")
    for i, ds in enumerate(selected_datasets, 1):
        print(f"  {i}. {ds.get('name', 'Unknown')[:70]}")

    print("\n" + "="*80)
    print("STARTING EXPORT")
    print("="*80)

    # Export each dataset
    results = []
    baltimore_count = 0

    for ds in selected_datasets:
        result = export_dataset(ds, output_dir)
        results.append(result)

        if result.get("success") and result.get("baltimore_info", {}).get("is_baltimore_dataset"):
            baltimore_count += 1

    # Summary
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)

    successful = sum(1 for r in results if r.get("success"))
    print(f"\n[SUCCESS] Exported {successful} out of {len(selected_datasets)} datasets")
    print(f"[INFO] Datasets with Baltimore/Maryland data: {baltimore_count}")

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
            readme += f"\n### {i}. {result['dataset_name']}\n"
            readme += f"- **UID:** {result['uid']}\n"
            readme += f"- **Organization:** {result.get('org', 'Unknown')}\n"
            readme += f"- **File:** `{result['filename']}`\n"
            readme += f"- **Rows:** {result['rows']:,}\n"

            balt_info = result.get('baltimore_info', {})
            if balt_info.get('is_baltimore_dataset'):
                readme += f"- **Maryland rows:** {balt_info.get('maryland_rows', 0):,}\n"
                readme += f"- **Baltimore rows:** {balt_info.get('baltimore_rows', 0):,}\n"
                readme += f"- **Contains Baltimore data:** YES ✓\n"
            else:
                readme += f"- **Contains Baltimore data:** No\n"
            readme += "\n"

    readme += """
---

## JSON File Structure:

Each JSON file contains:

```json
{
  "metadata": {
    "dataset_name": "Dataset Name",
    "uid": "unique-id",
    "organization": "CDC/SAMHSA",
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

# Access metadata
metadata = dataset['metadata']
print(f"Dataset: {metadata['dataset_name']}")
print(f"Rows: {metadata['total_rows']}")

# Convert to DataFrame
df = pd.DataFrame(dataset['data'])
print(df.head())

# Filter for Baltimore data (if applicable)
if metadata['baltimore_info']['is_baltimore_dataset']:
    # Find state/location columns
    for col in df.columns:
        if 'location' in col.lower() or 'state' in col.lower():
            baltimore_df = df[df[col].str.contains('Baltimore|MD', case=False, na=False)]
            print(f"Baltimore rows: {len(baltimore_df)}")
```

### JavaScript/Node.js:
```javascript
const fs = require('fs');

// Load dataset
const dataset = JSON.parse(fs.readFileSync('dataset_file.json', 'utf8'));

// Access data
console.log(`Dataset: ${dataset.metadata.dataset_name}`);
console.log(`Rows: ${dataset.metadata.total_rows}`);
console.log('Sample data:', dataset.data.slice(0, 5));

// Filter for Baltimore
if (dataset.metadata.baltimore_info.is_baltimore_dataset) {
    const baltimoreData = dataset.data.filter(row =>
        Object.values(row).some(val =>
            String(val).includes('Baltimore') || String(val).includes('MD')
        )
    );
    console.log(`Baltimore rows: ${baltimoreData.length}`);
}
```

---

## Filtering Baltimore/Maryland Data:

For datasets marked with Baltimore data:

**Common column names:**
- **State:** `State`, `StateAbbr`, `LocationDesc`, `StateDesc`
- **County:** `County`, `CountyName`, `LocationDesc`
- **FIPS codes:** Baltimore City = `24510`, Baltimore County = `24005`, Maryland = `24`

**Filter examples:**
```python
# Maryland state-level data
md_data = df[df['StateAbbr'] == 'MD']

# Baltimore specifically
baltimore = df[df['LocationDesc'].str.contains('Baltimore', case=False, na=False)]

# Using FIPS codes
baltimore_fips = df[df['CountyFIPS'].isin(['24510', '24005'])]
```

---

## Dataset Types:

Based on the datasets included, you'll find:
- **Behavioral Risk Factor Surveillance System (BRFSS)** - State/county health surveys
- **CDC Surveillance Data** - National and state-level health indicators
- **SAMHSA Data** - Substance abuse and mental health statistics
- **Youth Risk Behavior Survey (YRBSS)** - Student health data
- **Mental Health Indicators** - Various mental health metrics

---

**Ready to share with your team!**
**Use these datasets for analysis, visualization, or further development.**
"""

    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"\n[SUCCESS] Created: {output_dir}/README.md")
    print(f"\n[SUCCESS] All files ready in: {output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
