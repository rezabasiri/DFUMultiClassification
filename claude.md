# Claude Instructions

## Tracker Maintenance

When making changes to this repository, maintain the `tracker.md` file with the following guidelines:

### When to Update
Update `tracker.md` when making **major changes** to any file, including:
- Moving, renaming, or deleting files
- Significant refactoring or restructuring
- Adding new modules or major functionality
- Updating import statements across multiple files
- Modifying configuration or core utilities

### Format
Each entry should be:
- **Date**: YYYY-MM-DD format
- **Filename**: Specific file path (e.g., `src/main.py`)
- **Change**: Brief, specific description (1 line)

### Style Guidelines
- Be **specific** - mention exact filenames
- Be **brief** - one line per change
- Be **to the point** - no unnecessary details
- Use **action verbs** - "Moved", "Updated", "Created", "Refactored"

### Example Entry Format
```
## 2025-12-16
- **src/main.py**: Updated imports to use centralized config
- **src/utils/config.py**: Created centralized path configuration
- **data/**: Reorganized into raw/ and processed/ subdirectories
```

### What NOT to Track
- Minor bug fixes
- Small typo corrections
- Comment additions
- Trivial formatting changes

## Project Structure Notes

- Main training script: `src/main.py` (formerly `MultiModal_ModelV68_Combined_v1_31.py`)
- Main paper: `paper/main.tex`
- Configuration: `src/utils/config.py` - centralized path management
- Missing modules: See `archive/README.md` for list of referenced but missing files
