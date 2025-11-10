# Claude Code Skills Collection

This directory contains custom skills for Claude Code (claude.ai/code) to automate common development tasks.

## Available Skills

### 1. MD032-Fix: Blanks Around Lists

Automatically fix markdownlint MD032 violations by adding blank lines before lists.

**Command:** `/md032-fix`

**Documentation:** [md032-fix.md](md032-fix.md)

**Implementation:** [md032_fix_skill.py](md032_fix_skill.py)

**Usage Examples:**

```bash
# Fix all MD032 violations in current directory
python3 .claude/commands/md032_fix_skill.py .

# Fix MD032 violations in a specific directory
python3 .claude/commands/md032_fix_skill.py /path/to/markdown/files

# Check help
python3 .claude/commands/md032_fix_skill.py --help
```

**What it does:**

- Scans all `.md` files recursively
- Detects MD032 violations (missing blank lines before lists)
- Adds blank lines before lists
- Provides detailed report of fixes

**Example Output:**

```
Scanning for MD032 violations...
Directory: /Users/luqian/ScientificAI/TransferMatrix

✓ README.md: Fixed 5 MD032 issue(s)
  - Line 25: Added blank line before list
  - Line 42: Added blank line before list
  ...

✓ CONTRIBUTING.md: Fixed 2 MD032 issue(s)
  - Line 10: Added blank line before list
  - Line 18: Added blank line before list

Total: 7 MD032 issue(s) fixed across 2 file(s)
✅ All MD032 violations resolved!
```

**Features:**

- ✅ Safe operation (only adds whitespace, doesn't modify text)
- ✅ Idempotent (can run multiple times safely)
- ✅ Recursive directory scanning
- ✅ Detailed reporting
- ✅ Error handling

## Adding New Skills

To add a new skill:

1. Create a Python implementation in this directory
2. Add a markdown documentation file describing the skill
3. Update this README

## Skill Structure

Each skill should include:

- Python implementation (`.py` file)
- Markdown documentation (`.md` file) with:
  - Usage instructions
  - Description
  - Examples
  - Parameters/arguments
- Clear error handling
- Informative output

## Skill Design Principles

1. **Safe**: Don't modify content unnecessarily
2. **Idempotent**: Running multiple times should not cause issues
3. **Informative**: Provide clear feedback about actions taken
4. **Flexible**: Support common use cases
5. **Documented**: Include comprehensive examples and help text

## License

These skills are provided as-is for use with Claude Code.
