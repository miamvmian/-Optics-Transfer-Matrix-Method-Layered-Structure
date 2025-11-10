# Fix MD032: Blanks Around Lists

Automatically fix MD032 markdownlint violations (Lists should be surrounded by blank lines) in markdown files.

## Usage

```markdown
/md032-fix
```

## Description

This skill will:
1. Scan all markdown files in the current project for MD032 violations
2. Add blank lines before lists that don't have them
3. Provide a detailed report of fixes applied
4. Handle multiple markdown files in a single operation

## What it fixes

MD032 requires that lists be surrounded by blank lines. This skill adds blank lines **before** lists that are directly preceded by text.

**Before:**
```markdown
Text describing something.
- List item 1
- List item 2
```

**After:**
```markdown
Text describing something.

- List item 1
- List item 2
```

## Files it processes

By default, it scans and fixes:
- `*.md` files in the project directory
- Recursively processes all subdirectories
- Identifies markdown files and applies fixes

## Example output

```
Scanning for MD032 violations...

✓ /path/to/file1.md: Fixed 5 MD032 issues
  - Line 25: Added blank line before list
  - Line 42: Added blank line before list
  ...

✓ /path/to/file2.md: No issues found

Total: 5 MD032 issues fixed across 1 file(s)
✅ All MD032 violations resolved!
```

## Notes

- Only adds blank lines before lists (not after, as this is already handled by markdown parsers)
- Preserves existing blank lines
- Does not modify code blocks, headings, or other content
- Safe operation - only adds whitespace, doesn't change text content
- Can be run multiple times safely (idempotent)

## Related rules

This skill specifically addresses MD032. For other markdownlint rules (MD022, MD031, etc.), use appropriate skills or manual fixes.
