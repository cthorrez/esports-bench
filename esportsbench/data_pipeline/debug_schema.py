import os
import tempfile
import polars as pl

def find_first_bad_row(file_path):
    def create_temp_file(rows):
        with open(file_path, 'r') as src, tempfile.NamedTemporaryFile('w', delete=False) as dst:
            for _ in range(rows):
                line = src.readline()
                if not line:
                    break
                dst.write(line)
            return dst.name

    def create_single_row_temp(row_number):
        with open(file_path, 'r') as src:
            # Skip to target row
            for _ in range(row_number - 1):
                src.readline()
            line = src.readline()
            if not line:
                return None
            
            with tempfile.NamedTemporaryFile('w', delete=False) as dst:
                dst.write(line)
                return dst.name

    def test_rows(k):
        tmp_path = None
        try:
            tmp_path = create_temp_file(k)
            pl.read_ndjson(tmp_path)
            return True
        except (pl.exceptions.SchemaError, pl.exceptions.ComputeError):
            return False
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Get total rows
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines: {total_lines}")

    low, high = 1, total_lines
    first_bad = None

    while low <= high:
        mid = (low + high) // 2
        print(f"Testing first {mid} rows...", end=' ')
        
        if test_rows(mid):
            print("✓ Valid schema")
            low = mid + 1
        else:
            print("✗ Schema error")
            first_bad = mid
            high = mid - 1

    # Verification and schema printing
    if first_bad:
        print(f"\nVerifying error at row {first_bad}:")
        if first_bad == 1 or not test_rows(first_bad - 1):
            print("Validation failed - check initial rows")
            return None

        # Print schema of valid rows
        if first_bad > 1:
            print("\nSchema of valid rows (up to row {}):".format(first_bad - 1))
            valid_temp = create_temp_file(first_bad - 1)
            try:
                df_valid = pl.read_ndjson(valid_temp)
                print(df_valid.schema)
            except Exception as e:
                print("Error reading valid rows:", e)
            finally:
                os.unlink(valid_temp)
        else:
            print("\nNo valid rows before first row")

        # Print schema of problematic row
        print("\nSchema of problematic row (row {}):".format(first_bad))
        problematic_temp = create_single_row_temp(first_bad)
        if problematic_temp:
            try:
                # Try reading normally first
                df_problematic = pl.read_ndjson(problematic_temp)
                print("Explicit schema:", df_problematic.schema)
            except Exception as e:
                print("Error during explicit read:", e)
                # Fall back to lazy schema inference
                try:
                    lf = pl.scan_ndjson(problematic_temp)
                    print("Inferred schema:", lf.schema)
                except Exception as e:
                    print("Failed to infer schema:", e)
            finally:
                os.unlink(problematic_temp)
        else:
            print("Could not read problematic row")

        return first_bad

    return None

# Usage
file_path = "../../data/raw_data/dota2.jsonl"
bad_row = find_first_bad_row(file_path)
if bad_row:
    print(f"\nFirst schema inconsistency at row: {bad_row}")
    print(f"Examine line {bad_row} in: {file_path}")
else:
    print("\nNo schema errors found in entire file")