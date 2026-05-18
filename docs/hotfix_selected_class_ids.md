# Hotfix note: selected class filter array

## Problem

The previous runtime optimization patch replaced the per-frame expression:

```python
np.fromiter(selected_class_ids_set, dtype=int)
```

with a precomputed variable:

```python
selected_class_ids_arr
```

but in the tested local file the variable was not defined before use.

## Fix

Define the array once after:

```python
selected_class_ids_set = set(selected_idx or [])
```

Expected definition:

```python
selected_class_ids_arr = np.fromiter(selected_class_ids_set, dtype=int) if selected_class_ids_set else None
```

## Risk level

Low. The fix only affects class filtering before tracking. If no classes are selected, this branch is not used.
