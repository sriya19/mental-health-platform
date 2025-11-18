@echo off
echo ============================================================
echo EXPORTING 10 DATASETS TO JSON FORMAT
echo ============================================================
echo.
echo This script will copy 10 datasets from MinIO and convert them to JSON
echo.
echo Creating output directory...
mkdir team-datasets-json 2>nul

echo.
echo Copying datasets from MinIO container...

docker exec mental-health-platform-minio-1 sh -c "cd /data/mh-raw/raw && find . -name '*.parquet' -type f | head -10" > temp_files.txt

echo.
echo Starting Python export script...
python -c "
import subprocess
import json
import pandas as pd
from pathlib import Path

output_dir = Path('team-datasets-json')
output_dir.mkdir(exist_ok=True)

# Read list of files
with open('temp_files.txt', 'r') as f:
    files = [line.strip() for line in f if line.strip()]

print(f'Found {len(files)} parquet files to export')
print()

results = []
baltimore_count = 0

for i, file_path in enumerate(files[:10], 1):
    # Parse path: ./ORG/UID.parquet
    parts = file_path.strip('./').split('/')
    if len(parts) >= 2:
        org = parts[0]
        uid = parts[1].replace('.parquet', '')

        print(f'[{i}/10] Exporting {org}/{uid}...')

        try:
            # Copy file from container
            cmd = f'docker cp mental-health-platform-minio-1:/data/mh-raw/raw/{org}/{uid}.parquet temp.parquet'
            subprocess.run(cmd, shell=True, check=True, capture_output=True)

            # Read parquet
            df = pd.read_parquet('temp.parquet')

            # Check for Baltimore data
            cols_lower = [col.lower() for col in df.columns]
            has_baltimore = False
            md_rows = 0
            balt_rows = 0

            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['state', 'stateabbr', 'locationdesc']:
                    try:
                        md_rows = int(df[col].astype(str).str.contains('MD|Maryland', case=False, na=False, regex=True).sum())
                        if md_rows > 0:
                            has_baltimore = True
                            break
                    except:
                        pass

            if has_baltimore:
                print(f'    [BALTIMORE DATA FOUND] Maryland rows: {md_rows}')
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
                    'maryland_rows': md_rows
                },
                'data': json.loads(df.to_json(orient='records', date_format='iso'))
            }

            json_path = output_dir / json_filename
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False)

            file_size = json_path.stat().st_size / (1024 * 1024)
            print(f'    [SUCCESS] Exported: {json_filename} ({file_size:.2f} MB)')
            results.append({'success': True, 'org': org, 'uid': uid})

        except Exception as e:
            print(f'    [ERROR] Failed: {e}')
            results.append({'success': False, 'error': str(e)})

print()
print('='*60)
print(f'EXPORT COMPLETE')
print('='*60)
print(f'Exported: {sum(1 for r in results if r.get(\"success\"))} datasets')
print(f'Baltimore datasets: {baltimore_count}')
print(f'Output directory: team-datasets-json/')
print('='*60)

# Clean up temp file
import os
try:
    os.remove('temp.parquet')
    os.remove('temp_files.txt')
except:
    pass
"

echo.
echo ============================================================
echo DONE!
echo ============================================================
echo Check the team-datasets-json/ folder for your JSON files
echo.
pause
