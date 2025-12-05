#!/usr/bin/env python3
"""
Comprehensive Jupyter Notebook Validation Script
Validates exercise and solution notebooks for the Bread Financial Academy.

Usage:
    python validate_notebooks.py exercises/week_03/.../notebook.ipynb --type exercise
    python validate_notebooks.py solutions/week_03/.../notebook.ipynb --type solution
    python validate_notebooks.py --pair exercises/.../notebook.ipynb solutions/.../notebook.ipynb
    python validate_notebooks.py --requirements exercises/.../notebook.ipynb
"""

import json
import sys
import ast
import re
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Import name to package name mapping
IMPORT_TO_PACKAGE = {
    'sklearn': 'scikit-learn',
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'yaml': 'PyYAML',
    'dotenv': 'python-dotenv',
}


class NotebookValidator:
    """Main validator class for Jupyter notebooks."""

    def __init__(self, notebook_path: Path):
        self.path = notebook_path
        self.notebook = None
        self.cells = []
        self.errors = []
        self.warnings = []
        self.load_notebook()

    def load_notebook(self):
        """Load and parse the notebook JSON."""
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self.notebook = json.load(f)
                self.cells = self.notebook.get('cells', [])
        except Exception as e:
            self.errors.append(f"Failed to load notebook: {e}")

    def validate_syntax(self) -> bool:
        """Validate Python syntax in all code cells."""
        print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BLUE}Validating Python Syntax{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")

        syntax_errors = []

        for i, cell in enumerate(self.cells):
            if cell.get('cell_type') != 'code':
                continue

            cell_num = i + 1
            source = self._get_cell_source(cell)

            if not source.strip() or source.strip().startswith('!'):
                continue

            try:
                ast.parse(source)
                print(f"{Colors.GREEN}✅{Colors.RESET} Cell {cell_num}: Valid syntax")
            except SyntaxError as e:
                error_msg = f"Cell {cell_num}: Syntax error - {e.msg} at line {e.lineno}"
                syntax_errors.append(error_msg)
                print(f"{Colors.RED}❌{Colors.RESET} {error_msg}")
            except IndentationError as e:
                error_msg = f"Cell {cell_num}: Indentation error - {e.msg} at line {e.lineno}"
                syntax_errors.append(error_msg)
                print(f"{Colors.RED}❌{Colors.RESET} {error_msg}")
            except Exception:
                # Some edge cases are OK in notebooks (incomplete blocks, etc.)
                print(f"{Colors.YELLOW}⚠️{Colors.RESET}  Cell {cell_num}: Skipped (incomplete code block)")

        self.errors.extend(syntax_errors)

        if syntax_errors:
            print(f"\n{Colors.RED}Found {len(syntax_errors)} syntax errors{Colors.RESET}")
            return False
        else:
            print(f"\n{Colors.GREEN}✅ All code cells have valid syntax!{Colors.RESET}")
            return True

    def validate_exercise_placeholders(self) -> bool:
        """Validate that exercise notebook has proper = None placeholders in lab cells."""
        print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BLUE}Validating Exercise Placeholders{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")

        placeholder_pattern = re.compile(r'^\s*(\w+)\s*=\s*None\s*#.*YOUR CODE', re.MULTILINE)
        lab_cells_found = 0
        placeholder_counts = {}
        issues = []

        for i, cell in enumerate(self.cells):
            if cell.get('cell_type') != 'code':
                continue

            cell_num = i + 1
            source = self._get_cell_source(cell)

            # Check if this is a lab cell
            if 'Lab' in source and 'YOUR CODE' in source:
                lab_cells_found += 1
                placeholders = placeholder_pattern.findall(source)
                placeholder_counts[cell_num] = placeholders

                if placeholders:
                    print(f"{Colors.GREEN}✅{Colors.RESET} Cell {cell_num} (Lab cell): Found {len(placeholders)} placeholder(s)")
                    for var in placeholders:
                        print(f"    - {var}")
                else:
                    issue = f"Cell {cell_num}: Lab cell has no '= None' placeholders"
                    issues.append(issue)
                    print(f"{Colors.YELLOW}⚠️{Colors.RESET}  {issue}")

        if lab_cells_found == 0:
            warning = "No lab cells found (cells with 'Lab' and 'YOUR CODE')"
            self.warnings.append(warning)
            print(f"\n{Colors.YELLOW}⚠️  {warning}{Colors.RESET}")
            return True  # Not an error, just a warning

        self.errors.extend(issues)

        if issues:
            print(f"\n{Colors.RED}Found {len(issues)} issue(s) with placeholders{Colors.RESET}")
            return False
        else:
            print(f"\n{Colors.GREEN}✅ All {lab_cells_found} lab cells have proper placeholders!{Colors.RESET}")
            return True

    def validate_solution_completeness(self) -> bool:
        """Validate that solution notebook has no = None placeholders in lab cells."""
        print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BLUE}Validating Solution Completeness{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")

        # Pattern to find actual assignments to None (not in comments)
        none_assignment_pattern = re.compile(r'^\s*(\w+)\s*=\s*None\s*(?:#|$)', re.MULTILINE)
        lab_cells_found = 0
        issues = []

        for i, cell in enumerate(self.cells):
            if cell.get('cell_type') != 'code':
                continue

            cell_num = i + 1
            source = self._get_cell_source(cell)

            # Check if this is a lab cell (solution should have "SOLUTION" marker)
            if 'Lab' in source and 'SOLUTION' in source:
                lab_cells_found += 1

                # Check for any = None assignments
                none_assignments = []
                for line in source.split('\n'):
                    # Skip if line is a comment
                    if line.strip().startswith('#'):
                        continue
                    # Check for = None pattern
                    match = none_assignment_pattern.search(line)
                    if match:
                        none_assignments.append(match.group(1))

                if none_assignments:
                    issue = f"Cell {cell_num}: Solution cell still has '= None' assignments: {', '.join(none_assignments)}"
                    issues.append(issue)
                    print(f"{Colors.RED}❌{Colors.RESET} {issue}")
                else:
                    print(f"{Colors.GREEN}✅{Colors.RESET} Cell {cell_num} (Lab solution): Complete implementation")

        if lab_cells_found == 0:
            warning = "No lab solution cells found (cells with 'Lab' and 'SOLUTION')"
            self.warnings.append(warning)
            print(f"\n{Colors.YELLOW}⚠️  {warning}{Colors.RESET}")
            return True  # Not an error, just a warning

        self.errors.extend(issues)

        if issues:
            print(f"\n{Colors.RED}Found {len(issues)} incomplete solution(s){Colors.RESET}")
            return False
        else:
            print(f"\n{Colors.GREEN}✅ All {lab_cells_found} lab solutions are complete!{Colors.RESET}")
            return True

    def extract_imports(self) -> Set[str]:
        """Extract all import statements from the notebook."""
        imports = set()

        for cell in self.cells:
            if cell.get('cell_type') != 'code':
                continue

            source = self._get_cell_source(cell)

            for line in source.split('\n'):
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    imports.add(line)

        return imports

    def validate_imports(self) -> bool:
        """Validate that all imports are available."""
        print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BLUE}Validating Import Availability{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")

        imports = self.extract_imports()

        if not imports:
            print(f"{Colors.YELLOW}⚠️  No imports found in notebook{Colors.RESET}")
            return True

        print(f"Found {len(imports)} import statement(s)\n")

        failed_imports = []

        for import_stmt in sorted(imports):
            # Extract module name
            if import_stmt.startswith('import '):
                module = import_stmt.split()[1].split('.')[0].split(' as ')[0]
            elif import_stmt.startswith('from '):
                module = import_stmt.split()[1].split('.')[0]
            else:
                continue

            # Skip special modules
            if module in ['__future__']:
                continue

            # Try to import
            try:
                __import__(module)
                print(f"{Colors.GREEN}✅{Colors.RESET} {import_stmt}")
            except ImportError as e:
                print(f"{Colors.RED}❌{Colors.RESET} {import_stmt}")
                print(f"    Error: {e}")
                failed_imports.append(module)
            except Exception as e:
                print(f"{Colors.YELLOW}⚠️{Colors.RESET}  {import_stmt}")
                print(f"    Warning: {e}")

        if failed_imports:
            print(f"\n{Colors.RED}Missing modules: {', '.join(set(failed_imports))}{Colors.RESET}")
            self.errors.extend([f"Missing module: {m}" for m in failed_imports])
            return False
        else:
            print(f"\n{Colors.GREEN}✅ All imports available!{Colors.RESET}")
            return True

    def generate_requirements(self, output_path: Optional[Path] = None) -> List[str]:
        """Generate requirements.txt from notebook imports."""
        print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BLUE}Generating Requirements{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")

        imports = self.extract_imports()
        packages = set()

        for import_stmt in imports:
            # Extract module name
            if import_stmt.startswith('import '):
                module = import_stmt.split()[1].split('.')[0].split(' as ')[0]
            elif import_stmt.startswith('from '):
                module = import_stmt.split()[1].split('.')[0]
            else:
                continue

            # Skip standard library modules
            if module in ['sys', 'os', 'json', 're', 'time', 'pathlib', 'typing', 'ast',
                          'argparse', 'collections', 'itertools', 'functools', 'io']:
                continue

            # Map import name to package name
            package = IMPORT_TO_PACKAGE.get(module, module)
            packages.add(package)

        requirements = sorted(packages)

        print("Required packages:")
        for pkg in requirements:
            print(f"  - {pkg}")

        if output_path:
            with open(output_path, 'w') as f:
                f.write('\n'.join(requirements) + '\n')
            print(f"\n{Colors.GREEN}✅ Requirements written to {output_path}{Colors.RESET}")

        return requirements

    def _get_cell_source(self, cell: dict) -> str:
        """Extract source code from a cell."""
        source = cell.get('source', [])
        if isinstance(source, list):
            return ''.join(source)
        return source

    def print_summary(self):
        """Print validation summary."""
        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}Validation Summary{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")

        print(f"Notebook: {self.path.name}")
        print(f"Total cells: {len(self.cells)}")
        print(f"Code cells: {sum(1 for c in self.cells if c.get('cell_type') == 'code')}")
        print(f"Markdown cells: {sum(1 for c in self.cells if c.get('cell_type') == 'markdown')}")

        if self.errors:
            print(f"\n{Colors.RED}❌ Errors: {len(self.errors)}{Colors.RESET}")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print(f"\n{Colors.GREEN}✅ No errors found!{Colors.RESET}")

        if self.warnings:
            print(f"\n{Colors.YELLOW}⚠️  Warnings: {len(self.warnings)}{Colors.RESET}")
            for warning in self.warnings:
                print(f"  - {warning}")


def validate_paired_notebooks(exercise_path: Path, solution_path: Path) -> bool:
    """Validate that exercise and solution notebooks have matching structure."""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BLUE}Validating Paired Notebooks{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")

    # Load both notebooks
    with open(exercise_path, 'r') as f:
        exercise_nb = json.load(f)
    with open(solution_path, 'r') as f:
        solution_nb = json.load(f)

    exercise_cells = exercise_nb.get('cells', [])
    solution_cells = solution_nb.get('cells', [])

    issues = []

    # Check cell count
    if len(exercise_cells) != len(solution_cells):
        issue = f"Cell count mismatch: Exercise has {len(exercise_cells)}, Solution has {len(solution_cells)}"
        issues.append(issue)
        print(f"{Colors.RED}❌{Colors.RESET} {issue}")
    else:
        print(f"{Colors.GREEN}✅{Colors.RESET} Cell count matches: {len(exercise_cells)} cells")

    # Check cell types match
    min_cells = min(len(exercise_cells), len(solution_cells))
    type_mismatches = 0

    for i in range(min_cells):
        ex_type = exercise_cells[i].get('cell_type')
        sol_type = solution_cells[i].get('cell_type')

        if ex_type != sol_type:
            type_mismatches += 1
            issue = f"Cell {i+1}: Type mismatch (Exercise: {ex_type}, Solution: {sol_type})"
            issues.append(issue)
            print(f"{Colors.RED}❌{Colors.RESET} {issue}")

    if type_mismatches == 0:
        print(f"{Colors.GREEN}✅{Colors.RESET} All cell types match")
    else:
        print(f"{Colors.RED}❌{Colors.RESET} Found {type_mismatches} cell type mismatches")

    # Summary
    if issues:
        print(f"\n{Colors.RED}Paired validation FAILED: {len(issues)} issue(s){Colors.RESET}")
        return False
    else:
        print(f"\n{Colors.GREEN}✅ Paired validation PASSED!{Colors.RESET}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Validate Jupyter notebooks for Bread Financial Academy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate exercise notebook
  python validate_notebooks.py exercises/week_03/.../notebook.ipynb --type exercise

  # Validate solution notebook
  python validate_notebooks.py solutions/week_03/.../notebook.ipynb --type solution

  # Validate paired notebooks
  python validate_notebooks.py --pair exercises/.../notebook.ipynb solutions/.../notebook.ipynb

  # Generate requirements.txt
  python validate_notebooks.py exercises/.../notebook.ipynb --requirements
        """
    )

    parser.add_argument('notebook', nargs='?', type=Path, help='Path to notebook file')
    parser.add_argument('--type', choices=['exercise', 'solution'], help='Notebook type to validate')
    parser.add_argument('--pair', nargs=2, type=Path, metavar=('EXERCISE', 'SOLUTION'),
                        help='Validate paired exercise and solution notebooks')
    parser.add_argument('--requirements', action='store_true', help='Generate requirements.txt')

    args = parser.parse_args()

    # Paired validation
    if args.pair:
        exercise_path, solution_path = args.pair

        if not exercise_path.exists():
            print(f"{Colors.RED}❌ Exercise notebook not found: {exercise_path}{Colors.RESET}")
            sys.exit(1)
        if not solution_path.exists():
            print(f"{Colors.RED}❌ Solution notebook not found: {solution_path}{Colors.RESET}")
            sys.exit(1)

        success = validate_paired_notebooks(exercise_path, solution_path)
        sys.exit(0 if success else 1)

    # Single notebook validation
    if not args.notebook:
        parser.print_help()
        sys.exit(1)

    if not args.notebook.exists():
        print(f"{Colors.RED}❌ Notebook not found: {args.notebook}{Colors.RESET}")
        sys.exit(1)

    validator = NotebookValidator(args.notebook)

    if validator.errors:
        print(f"{Colors.RED}❌ Failed to load notebook{Colors.RESET}")
        sys.exit(1)

    # Run validations based on type
    all_passed = True

    # Always validate syntax and imports
    all_passed &= validator.validate_syntax()
    all_passed &= validator.validate_imports()

    # Type-specific validations
    if args.type == 'exercise':
        all_passed &= validator.validate_exercise_placeholders()
    elif args.type == 'solution':
        all_passed &= validator.validate_solution_completeness()

    # Generate requirements if requested
    if args.requirements:
        req_path = args.notebook.parent / 'requirements.txt'
        validator.generate_requirements(req_path)

    # Print summary
    validator.print_summary()

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
