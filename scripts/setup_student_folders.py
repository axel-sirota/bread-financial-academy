#!/usr/bin/env python3
"""
Setup Student Folders in SageMaker Studio Shared Workspace

This script creates individual folders for all 66 students + admin in the shared workspace.
Run this ONCE after the shared JupyterLab app is created for the first time.

Usage:
    python3 setup_student_folders.py

Location: Run from JupyterLab terminal in shared-academy-workspace
"""

import os
import sys
from pathlib import Path

# Number of students
NUM_STUDENTS = 66

# Admin folder name
ADMIN_FOLDER = "admin-axel"

# Base directory (current working directory in JupyterLab)
BASE_DIR = Path.cwd()

def create_student_folders():
    """Create folders for admin + student1 through student66"""

    print(f"Creating admin folder + {NUM_STUDENTS} student folders in: {BASE_DIR}")
    print("-" * 60)

    created_count = 0
    skipped_count = 0

    # Create admin folder first
    admin_path = BASE_DIR / ADMIN_FOLDER
    try:
        if admin_path.exists():
            print(f"✓ {ADMIN_FOLDER}/ already exists - skipping")
            skipped_count += 1
        else:
            admin_path.mkdir(mode=0o755, exist_ok=True)

            readme_path = admin_path / "README.md"
            readme_content = f"""# {ADMIN_FOLDER} Workspace

This folder is for the **admin** to organize teaching materials and demos.

## Contents

- Demo notebooks for live coding sessions
- Solution templates
- Testing notebooks
- Admin utilities

## Tips

- Use this folder to prepare materials before class
- Test notebooks here before sharing with students
- Keep backup copies of important demos

Happy teaching! 🎓
"""
            readme_path.write_text(readme_content)

            print(f"✓ Created {ADMIN_FOLDER}/ with README.md")
            created_count += 1

    except Exception as e:
        print(f"✗ Error creating {ADMIN_FOLDER}/: {e}", file=sys.stderr)
        return False

    # Create student folders
    for i in range(1, NUM_STUDENTS + 1):
        folder_name = f"student{i}"
        folder_path = BASE_DIR / folder_name

        try:
            if folder_path.exists():
                print(f"✓ {folder_name}/ already exists - skipping")
                skipped_count += 1
            else:
                folder_path.mkdir(mode=0o755, exist_ok=True)

                # Create a README.md in each folder
                readme_path = folder_path / "README.md"
                readme_content = f"""# {folder_name} Workspace

This folder is for **{folder_name}** to organize their notebooks and files.

## Recommended File Naming Convention

- Week 5: `{folder_name}_week5_ai_services.ipynb`
- Week 6: `{folder_name}_week6_neural_networks.ipynb`
- Week 7: `{folder_name}_week7_mlflow_monitoring.ipynb`

## Tips

- Save your work frequently (Ctrl+S or Cmd+S)
- Upload notebooks here before starting class
- Download completed notebooks after class for backup
- Use descriptive filenames for your experiments

Happy learning! 🚀
"""
                readme_path.write_text(readme_content)

                print(f"✓ Created {folder_name}/ with README.md")
                created_count += 1

        except Exception as e:
            print(f"✗ Error creating {folder_name}/: {e}", file=sys.stderr)
            return False

    print("-" * 60)
    print(f"\n✅ Folder setup complete!")
    print(f"   - Created: {created_count} folders")
    print(f"   - Skipped: {skipped_count} folders (already existed)")
    print(f"\nFolders created:")
    print(f"   - Admin: {ADMIN_FOLDER}/")
    print(f"   - Students: student1/, student2/, ..., student{NUM_STUDENTS}/")

    return True

def verify_environment():
    """Verify we're running in the right environment"""

    # Check if we're likely in SageMaker Studio
    if not os.path.exists('/opt/ml'):
        print("⚠️  Warning: This doesn't appear to be a SageMaker environment")
        print("   Expected to find /opt/ml directory")

        response = input("\nContinue anyway? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Aborted.")
            return False

    # Check write permissions
    test_file = BASE_DIR / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        print(f"✗ Error: No write permissions in {BASE_DIR}")
        print(f"   {e}")
        return False

    return True

def main():
    """Main execution"""

    print("=" * 60)
    print("SageMaker Studio Shared Workspace - Student Folder Setup")
    print("=" * 60)
    print()

    # Verify environment
    if not verify_environment():
        sys.exit(1)

    # Create folders
    if not create_student_folders():
        sys.exit(1)

    print("\n🎓 All 67 users (admin + 66 students) can now use their assigned folders!")
    sys.exit(0)

if __name__ == "__main__":
    main()
