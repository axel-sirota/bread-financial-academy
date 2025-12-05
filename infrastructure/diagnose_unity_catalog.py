"""
Unity Catalog Diagnostic Script for Databricks

Run this in a Databricks notebook cell to diagnose volume access issues.
Copy and paste the code below into a Databricks notebook cell.
"""

DIAGNOSTIC_CODE = '''
# Unity Catalog Diagnostic Script
# Run this in a Databricks notebook to check what exists

print("="*80)
print("UNITY CATALOG DIAGNOSTIC")
print("="*80)

# Check 1: List available catalogs
print("\\n1. Available Catalogs:")
print("-"*80)
try:
    catalogs = spark.sql("SHOW CATALOGS").collect()
    for row in catalogs:
        print(f"   ✅ {row.catalog}")
except Exception as e:
    print(f"   ❌ Error: {str(e)[:200]}")

# Check 2: Check if bread_financial_academy catalog exists
print("\\n2. Checking bread_financial_academy catalog:")
print("-"*80)
try:
    schemas = spark.sql("SHOW SCHEMAS IN bread_financial_academy").collect()
    print(f"   ✅ Catalog exists! Found {len(schemas)} schemas:")
    for row in schemas:
        print(f"      - {row.databaseName}")
except Exception as e:
    print(f"   ❌ Catalog not found or no access")
    print(f"   Error: {str(e)[:200]}")
    print("\\n   SOLUTION: Ask IT to create catalog:")
    print("   SQL: CREATE CATALOG IF NOT EXISTS bread_financial_academy;")

# Check 3: Check if shared_datasets schema exists
print("\\n3. Checking shared_datasets schema:")
print("-"*80)
try:
    volumes = spark.sql("SHOW VOLUMES IN bread_financial_academy.shared_datasets").collect()
    print(f"   ✅ Schema exists! Found {len(volumes)} volumes:")
    for row in volumes:
        print(f"      - {row.volume_name}")
except Exception as e:
    print(f"   ❌ Schema not found or no access")
    print(f"   Error: {str(e)[:200]}")
    print("\\n   SOLUTION: Ask IT to create schema:")
    print("   SQL: CREATE SCHEMA IF NOT EXISTS bread_financial_academy.shared_datasets;")

# Check 4: Check if datasets volume exists
print("\\n4. Checking datasets volume:")
print("-"*80)
try:
    volume_info = spark.sql("""
        DESCRIBE VOLUME bread_financial_academy.shared_datasets.datasets
    """).collect()
    print(f"   ✅ Volume exists!")
    for row in volume_info:
        print(f"      {row.info_name}: {row.info_value}")
except Exception as e:
    print(f"   ❌ Volume not found or no access")
    print(f"   Error: {str(e)[:200]}")
    print("\\n   SOLUTION: Ask IT to create volume:")
    print("   SQL: CREATE VOLUME IF NOT EXISTS bread_financial_academy.shared_datasets.datasets;")

# Check 5: Check current user and groups
print("\\n5. Current User & Groups:")
print("-"*80)
try:
    user = spark.sql("SELECT current_user() as user").collect()[0].user
    print(f"   User: {user}")
except Exception as e:
    print(f"   ❌ Cannot get user: {str(e)[:100]}")

try:
    groups = spark.sql("SHOW GROUPS").collect()
    print(f"   Groups: {', '.join([row.groupName for row in groups])}")
except Exception as e:
    print(f"   ❌ Cannot list groups: {str(e)[:100]}")

# Check 6: Try alternative volume paths
print("\\n6. Testing alternative volume access patterns:")
print("-"*80)

test_paths = [
    "/Volumes/bread_financial_academy/shared_datasets/datasets",
    "/Volumes/bread_financial_academy/shared_datasets",
    "/Volumes/bread_financial_academy",
]

for path in test_paths:
    try:
        files = dbutils.fs.ls(path)
        print(f"   ✅ {path} - ACCESSIBLE ({len(files)} items)")
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"   ❌ {path} - NOT ACCESSIBLE")
        print(f"      Error: {error_msg}")

# Check 7: Recommended fix
print("\\n" + "="*80)
print("RECOMMENDED ACTIONS:")
print("="*80)
print("""
If the volume doesn't exist, ask your IT team to run these SQL commands:

-- 1. Create catalog (if it doesn't exist)
CREATE CATALOG IF NOT EXISTS bread_financial_academy;

-- 2. Create schema (if it doesn't exist)
CREATE SCHEMA IF NOT EXISTS bread_financial_academy.shared_datasets;

-- 3. Create volume (if it doesn't exist)
CREATE VOLUME IF NOT EXISTS bread_financial_academy.shared_datasets.datasets;

-- 4. Grant permissions to DS_Academy group
GRANT USE CATALOG ON CATALOG bread_financial_academy TO `DS_Academy`;
GRANT USE SCHEMA ON SCHEMA bread_financial_academy.shared_datasets TO `DS_Academy`;
GRANT READ VOLUME, WRITE VOLUME ON VOLUME bread_financial_academy.shared_datasets.datasets TO `DS_Academy`;

-- 5. Add users to DS_Academy group
-- Replace 'student@example.com' with actual student emails
ALTER GROUP `DS_Academy` ADD MEMBER 'student@example.com';

""")
print("="*80)
'''

if __name__ == "__main__":
    print("Copy the code below and paste into a Databricks notebook cell:")
    print("\n" + "="*80 + "\n")
    print(DIAGNOSTIC_CODE)
