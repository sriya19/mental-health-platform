"""
Export 10 datasets to JSON - Direct MinIO access
"""
import subprocess
import json
import pandas as pd
from pathlib import Path
import os

def main():
    print("="*80)
    print("EXPORTING 10 DATASETS TO JSON FORMAT")
    print("="*80)

    output_dir = Path('team-datasets-json')
    output_dir.mkdir(exist_ok=True)
    print(f"\n[INFO] Output directory: {output_dir}/\n")

    # List parquet files from MinIO container
    print("[INFO] Listing files from MinIO...")
    cmd = 'docker exec mental-health-platform-minio-1 find /data/mh-raw/raw -name "*.parquet" -type f'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Could not list MinIO files: {result.stderr}")
        return

    files = [line.strip() for line in result.stdout.strip().split('\n') if line.strip() and '/.trash' not in line]

    print(f"[INFO] Found {len(files)} parquet files")

    # Prioritize BRFSS (likely has Baltimore data)
    brfss_files = [f for f in files if '/BRFSS/' in f]
    other_files = [f for f in files if '/BRFSS/' not in f]

    sorted_files = brfss_files + other_files
    selected_files = sorted_files[:10]

    print(f"[INFO] Selected 10 datasets ({len(brfss_files)} BRFSS)")
    print("\n" + "="*80)
    print("STARTING EXPORT")
    print("="*80)

    results = []
    baltimore_count = 0

    for i, file_path in enumerate(selected_files, 1):
        # Parse path: /data/mh-raw/raw/ORG/UID.parquet
        parts = file_path.split('/')
        if len(parts) < 2:
            continue

        org = parts[-2]
        uid = parts[-1].replace('.parquet', '')

        print(f"\n[{i}/10] EXPORT {org}/{uid}")

        try:
            # Copy file from container to temp location
            temp_file = 'temp_export.parquet'
            cmd = f'docker cp mental-health-platform-minio-1:{file_path} {temp_file}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"         [ERROR] Copy failed: {result.stderr}")
                continue

            # Read parquet
            df = pd.read_parquet(temp_file)
            print(f"         Rows: {len(df):,} | Columns: {len(df.columns)}")

            # Check for Baltimore data
            has_baltimore = False
            md_rows = 0
            balt_rows = 0

            cols_lower = [col.lower() for col in df.columns]
            state_cols = ['state', 'stateabbr', 'locationdesc', 'statedesc']

            for col in df.columns:
                if col.lower() in state_cols:
                    try:
                        md_rows = int(df[col].astype(str).str.contains('MD|Maryland', case=False, na=False, regex=True).sum())
                        if md_rows > 0:
                            has_baltimore = True
                            break
                    except:
                        pass

            if 'county' in ' '.join(cols_lower) or 'location' in ' '.join(cols_lower):
                for col in df.columns:
                    if any(x in col.lower() for x in ['county', 'location']):
                        try:
                            balt_rows = int(df[col].astype(str).str.contains('Baltimore', case=False, na=False).sum())
                            if balt_rows > 0:
                                has_baltimore = True
                                break
                        except:
                            pass

            if has_baltimore:
                print(f"         [BALTIMORE DATA FOUND!]")
                print(f"         Maryland: {md_rows:,} rows | Baltimore: {balt_rows:,} rows")
                baltimore_count += 1

            # Create JSON
            json_filename = f'{org}_{uid}.json'
            data_dict = {
                'metadata': {
                    'dataset_name': f'{org}:{uid}',
                    'uid': uid,
                    'organization': org,
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'columns': list(df.columns),
                    'has_baltimore_data': has_baltimore,
                    'maryland_rows': md_rows,
                    'baltimore_rows': balt_rows
                },
                'data': json.loads(df.to_json(orient='records', date_format='iso'))
            }

            json_path = output_dir / json_filename
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False)

            file_size = json_path.stat().st_size / (1024 * 1024)
            print(f"         [SUCCESS] Exported: {json_filename} ({file_size:.2f} MB)")

            results.append({
                'success': True,
                'org': org,
                'uid': uid,
                'filename': json_filename,
                'has_baltimore': has_baltimore
            })

            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)

        except Exception as e:
            print(f"         [ERROR] Failed: {e}")
            results.append({'success': False, 'error': str(e)})

    # Summary
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)

    successful = sum(1 for r in results if r.get('success'))
    print(f"\n[SUCCESS] Exported {successful} out of {len(selected_files)} datasets")
    print(f"[INFO] Datasets with Baltimore/Maryland data: {baltimore_count}")

    # Calculate total size
    json_files = list(output_dir.glob("*.json"))
    total_size = sum(f.stat().st_size for f in json_files) / (1024 * 1024)

    print(f"\n[SUMMARY]")
    print(f"  Files exported: {len(json_files)}")
    print(f"  Total size: {total_size:.2f} MB")
    print(f"  Baltimore datasets: {baltimore_count}")
    print(f"\n[SUCCESS] All files in: {output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
