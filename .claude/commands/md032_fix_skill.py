#!/usr/bin/env python3
"""
MD032-Fix Skill: Automatically fix markdownlint MD032 violations (blanks around lists)

This script scans markdown files and adds blank lines before lists that don't have them,
fixing MD032 violations (Lists should be surrounded by blank lines).
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict


class MD032Fixer:
    """Fix MD032 violations in markdown files."""

    def __init__(self):
        """Initialize the MD032 fixer."""
        self.fixed_files = []
        self.errors = []

    def find_markdown_files(self, directory: str = ".") -> List[str]:
        """Find all markdown files in a directory recursively.

        Args:
            directory: Directory to search in (default: current directory)

        Returns:
            List of paths to markdown files
        """
        md_files = []
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                if file.endswith(".md") and not file.startswith("."):
                    md_files.append(os.path.join(root, file))

        return md_files

    def find_md032_violations(self, filepath: str) -> List[int]:
        """Find line numbers with MD032 violations (missing blank before list).

        Args:
            filepath: Path to markdown file

        Returns:
            List of line numbers (0-indexed) where violations occur
        """
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        violations = []
        for i in range(len(lines) - 1):
            line = lines[i]
            next_line = lines[i + 1] if i + 1 < len(lines) else ""

            # Check if current line ends with text and next line is a list
            if (
                line.strip()
                and not line.strip().startswith("#")
                and not line.strip().startswith("```")
                and not line.strip().endswith(":")  # Skip if ends with colon
                and not re.match(r"^\s*[-*+]\s", line)
                and not re.match(r"^\s*\d+\.\s", line)
                and not re.match(r"^\s*\[\s*[ xX]\]\s", line)
            ):
                # Check if next line is a list item
                if (
                    next_line.strip().startswith("- ")
                    or next_line.strip().startswith("* ")
                    or next_line.strip().startswith("+ ")
                    or any(
                        next_line.strip().startswith(f"{n}. ") for n in range(1, 100)
                    )
                    or next_line.strip().startswith("- **")
                    or next_line.strip().startswith("- `")
                ):
                    violations.append(i)

        return violations

    def fix_file(self, filepath: str) -> Tuple[int, List[int]]:
        """Fix MD032 violations in a single file.

        Args:
            filepath: Path to markdown file

        Returns:
            Tuple of (number of violations fixed, list of line numbers fixed)
        """
        violations = self.find_md032_violations(filepath)

        if not violations:
            return 0, []

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        fixed_lines = []
        i = 0
        fixed_at_lines = []

        while i < len(lines):
            line = lines[i]
            next_line = lines[i + 1] if i + 1 < len(lines) else None

            # Check if this is a violation line
            if i in violations and next_line is not None:
                # Add current line
                fixed_lines.append(line)
                # Add blank line
                fixed_lines.append("")
                # Add next line (the list)
                fixed_lines.append(next_line)
                fixed_at_lines.append(i + 1)  # Store 1-indexed line number
                i += 2
            else:
                fixed_lines.append(line)
                i += 1

        # Write back
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(fixed_lines))
            # Ensure file ends with newline
            if not content.endswith("\n"):
                f.write("\n")

        self.fixed_files.append(filepath)
        return len(violations), fixed_at_lines

    def fix_all(self, directory: str = ".") -> Dict[str, Tuple[int, List[int]]]:
        """Fix all MD032 violations in markdown files.

        Args:
            directory: Directory to process (default: current directory)

        Returns:
            Dictionary mapping file paths to (count, line_numbers) tuples
        """
        md_files = self.find_markdown_files(directory)
        results = {}

        for filepath in md_files:
            try:
                count, lines = self.fix_file(filepath)
                if count > 0:
                    results[filepath] = (count, lines)
            except Exception as e:
                self.errors.append(f"{filepath}: {str(e)}")

        return results

    def print_report(self, results: Dict[str, Tuple[int, List[int]]]):
        """Print a detailed report of fixes applied.

        Args:
            results: Dictionary from fix_all() method
        """
        print("\n" + "=" * 70)
        print("MD032 Fix Report")
        print("=" * 70)

        if self.errors:
            print("\n⚠️  Errors encountered:")
            for error in self.errors:
                print(f"  {error}")

        if not results:
            print("\n✅ No MD032 violations found!")
            return

        print(f"\nFixed {len(results)} file(s):\n")

        total_fixed = 0
        for filepath, (count, lines) in results.items():
            rel_path = os.path.relpath(filepath)
            print(f"✓ {rel_path}: Fixed {count} MD032 issue(s)")
            for line in lines:
                print(f"  - Line {line}: Added blank line before list")
            total_fixed += count
            print()

        print("=" * 70)
        print(
            f"Total: {total_fixed} MD032 issue(s) fixed across {len(results)} file(s)"
        )
        print("=" * 70)
        print("\n✅ All MD032 violations resolved!")


def main():
    """Main entry point for the MD032-fix skill."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix MD032 markdownlint violations (blanks around lists)"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to process (default: current directory)",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    args = parser.parse_args()

    # Check if directory exists
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' not found", file=sys.stderr)
        sys.exit(1)

    print("Scanning for MD032 violations...")
    print(f"Directory: {args.directory}\n")

    fixer = MD032Fixer()
    results = fixer.fix_all(args.directory)
    fixer.print_report(results)


if __name__ == "__main__":
    main()
