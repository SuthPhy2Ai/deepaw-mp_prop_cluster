# Documentation Index

This directory contains all project documentation organized by category.

## Directory Structure

```
docs/
├── deepaw_integration/     # DeePAW feature integration documentation
├── guides/                 # User guides and tutorials
└── project_status/         # Project status and progress tracking
```

---

## DeePAW Integration (`deepaw_integration/`)

Documentation for integrating DeePAW pretrained atomic features into the MP pipeline.

- **[README.md](deepaw_integration/README.md)** - Complete technical documentation
  - Architecture design
  - Implementation details
  - Usage instructions
  - Performance expectations

- **[QUICK_START.md](deepaw_integration/QUICK_START.md)** - 5-minute quick start guide
  - Three ways to use DeePAW features
  - Common parameters
  - Troubleshooting

- **[IMPLEMENTATION_NOTES.md](deepaw_integration/IMPLEMENTATION_NOTES.md)** - Implementation notes
  - Design decisions (Plan B rationale)
  - Problems encountered and solutions
  - Performance analysis
  - Future improvements

- **[deepaw_atom_tower_integration.md](deepaw_integration/deepaw_atom_tower_integration.md)** - Original integration plan

- **[DEEPAW_LOCAL_SETUP.md](deepaw_integration/DEEPAW_LOCAL_SETUP.md)** - Local environment setup

---

## Guides (`guides/`)

User guides and tutorials for working with the project.

- **[PHASE2_QUICKSTART.md](guides/PHASE2_QUICKSTART.md)** - Phase 2 enhanced backbone quick start

---

## Project Status (`project_status/`)

Project progress tracking and status reports.

- **[PHASE1_DONE.txt](project_status/PHASE1_DONE.txt)** - Phase 1 completion summary
- **[PHASE1_PHASE2_CHECKLIST.md](project_status/PHASE1_PHASE2_CHECKLIST.md)** - Phase 1 & 2 checklist
- **[README_STATUS.md](project_status/README_STATUS.md)** - Overall project status

---

## Root Documentation

Files in the project root:

- **[README.md](../README.md)** - Main project README
- **[CLAUDE.md](../CLAUDE.md)** - Instructions for Claude Code
- **[requirements.txt](../requirements.txt)** - Python dependencies

---

## Quick Links

### Getting Started
- [Main README](../README.md)
- [Phase 2 Quick Start](guides/PHASE2_QUICKSTART.md)
- [DeePAW Quick Start](deepaw_integration/QUICK_START.md)

### Implementation Details
- [DeePAW Integration](deepaw_integration/README.md)
- [Implementation Notes](deepaw_integration/IMPLEMENTATION_NOTES.md)

### Project Status
- [Phase 1 Complete](project_status/PHASE1_DONE.txt)
- [Project Checklist](project_status/PHASE1_PHASE2_CHECKLIST.md)

---

## Contributing Documentation

When adding new documentation:

1. **Choose the right category**:
   - Integration docs → `deepaw_integration/` or create new integration folder
   - User guides → `guides/`
   - Status reports → `project_status/`

2. **Update this index** after adding new files

3. **Use clear naming**:
   - `README.md` for main documentation
   - `QUICK_START.md` for getting started guides
   - `IMPLEMENTATION_NOTES.md` for technical details
   - Descriptive names for specific topics

4. **Cross-reference** related documents using relative links
