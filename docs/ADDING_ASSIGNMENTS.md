# Adding Future Assignments

Use the same pattern for every new series so the repository remains easy to scan from a resume.

## Folder template

```text
assignments/
└── assignment-03/
    ├── README.md
    ├── 01-descriptive-topic-name/
    │   ├── input/
    │   ├── output/
    │   ├── algorithm.py
    │   ├── experiment.py
    │   └── test.py
    └── 02-another-topic/
        └── ...
```

## Checklist

1. Use descriptive directory names instead of only `Q1`, `Q2`, etc.
2. Keep the main algorithm separate from plotting/experiment code when practical.
3. Keep every experiment reproducible from local inputs.
4. Add a standalone `test.py` for algorithmic and output checks.
5. Add one or two representative figures to the assignment README.
6. Put quantitative results in a small Markdown table.
7. Register the new demo/test pair in `scripts/validate_repository.py`.
8. Avoid publishing student IDs, private submission metadata, or full course materials.
9. If an input asset has unclear redistribution rights, replace it with a licensed/open alternative before making the repository public.
