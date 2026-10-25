"""
Name-based class bookkeeping of the SNGP confidence model

The notebooks identified classes by integer indices, which shift whenever the checkpoint's
dataset list or the additional datasets change. Here every operation takes class *names*
(TOML `RemoveClasses`, `MergeGroups`, `PosthocGroups`, `DisplayNames`), is validated, and
reproduces the notebook numbering (and therefore the shipped metadata):

1. combined classes: baseline dataset keys in checkpoint order, then the alphabetically
   sorted case names of the additional datasets;
2. removal: survivors renumbered in increasing old-label order;
3. merge: groups first (TOML order), then the ungrouped classes in increasing label order.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class ClassMap:
    """
    Ordered class names; the label of a class is its position

    Attributes:
        names: class names in label order
        index: name -> label
    """
    names: List[str]
    index: Dict[str, int] = field(init=False)

    def __post_init__(self):
        self.names = [str(n) for n in self.names]
        if len(set(self.names)) != len(self.names):
            dup = sorted({n for n in self.names if self.names.count(n) > 1})
            raise ValueError(f"Duplicate class names: {dup}")
        self.index = {n: i for i, n in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def to_dict(self) -> Dict[str, int]:
        """name -> label dict (the metadata 'flow_to_label_*' format)."""
        return dict(self.index)

    @classmethod
    def from_dict(cls, flow_to_label: Dict[str, int]) -> "ClassMap":
        """
        Build from a metadata 'flow_to_label_*' dict

        Args:
            flow_to_label: name -> label, labels must be 0..K-1

        Returns:
            ClassMap
        """
        items = sorted(flow_to_label.items(), key=lambda kv: int(kv[1]))
        labels = [int(v) for _, v in items]
        if labels != list(range(len(labels))):
            raise ValueError(f"Class labels are not contiguous 0..{len(labels) - 1}: {labels}")
        return cls([k for k, _ in items])

    def ids(self, names: Sequence[str], what: str = "class") -> List[int]:
        """
        Labels of class names (validated)

        Args:
            names: class names
            what: wording of the error message

        Returns:
            list of labels
        """
        unknown = [n for n in names if n not in self.index]
        if unknown:
            raise ValueError(f"Unknown {what} name(s) {unknown}. Valid names: {self.names}")
        return [self.index[n] for n in names]


def combined_class_map(baseline_keys: Sequence[str], extra_case_names: np.ndarray) -> ClassMap:
    """
    Classes of the baseline datasets followed by the additional datasets

    Args:
        baseline_keys: Baseline training dataset keys in checkpoint order (training_dataset_keys)
        extra_case_names: Per-point case names of the additional datasets (any order)

    Returns:
        ClassMap: baseline keys, then np.unique(extra case names) (alphabetical), as in the notebooks
    """
    base = [str(k) for k in baseline_keys]
    extra = [str(n) for n in np.unique(np.asarray(extra_case_names).astype(str))] if len(extra_case_names) else []
    clash = sorted(set(base) & set(extra))
    if clash:
        raise ValueError(f"Additional-dataset case names {clash} collide with baseline dataset keys; "
                         "the classes would be merged silently")
    return ClassMap(base + extra)


def labels_from_names(sample_names: np.ndarray, class_map: ClassMap) -> np.ndarray:
    """
    Integer labels of per-sample class names

    Args:
        sample_names: (N,) names
        class_map: ClassMap

    Returns:
        (N,) int64 labels
    """
    names = np.asarray(sample_names).astype(str)
    uniq, inv = np.unique(names, return_inverse=True)
    lut = np.array(class_map.ids(uniq.tolist(), what="sample class"), dtype=np.int64)
    return lut[inv].reshape(-1)


def labels_to_remove(class_map: ClassMap, remove: Sequence[str]) -> List[int]:
    """
    Sorted labels of removed classes (metadata 'labels_to_remove', combined numbering)

    Args:
        class_map: Combined ClassMap
        remove: Class names to remove

    Returns:
        Sorted list of int labels
    """
    return sorted(class_map.ids(list(remove), what="RemoveClasses"))


def remove_classes(labels: np.ndarray, class_map: ClassMap,
                   remove: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, ClassMap]:
    """
    Drop the points of some classes and renumber the rest

    The renumbering follows the class map (survivors in increasing old-label order), not the
    labels present in `labels`, so train and valid subsets always get the same numbering.

    Args:
        labels: (N,) labels in `class_map` numbering
        class_map: ClassMap of `labels`
        remove: Class names to drop

    Returns:
        (mask of kept points, new labels of the kept points, new ClassMap)
    """
    removed = set(class_map.ids(list(remove), what="RemoveClasses"))
    old_to_new = np.full(len(class_map), -1, dtype=np.int64)
    kept_names = []
    for old, name in enumerate(class_map.names):
        if old not in removed:
            old_to_new[old] = len(kept_names)
            kept_names.append(name)
    labels = np.asarray(labels, dtype=np.int64)
    mask = ~np.isin(labels, list(removed))
    return mask, old_to_new[labels[mask]], ClassMap(kept_names)


def _group_mapping(class_map: ClassMap, groups: Dict[str, List[str]], what: str) -> Tuple[np.ndarray, List[str]]:
    """old label -> new label for `groups` (groups first, then ungrouped in label order)."""
    assigned: Dict[str, str] = {}
    for gname, members in groups.items():
        if not members:
            raise ValueError(f"{what}: group '{gname}' is empty")
        class_map.ids(members, what=f"{what} member")
        for m in members:
            if m in assigned:
                raise ValueError(f"{what}: class '{m}' is in groups '{assigned[m]}' and '{gname}'")
            assigned[m] = gname
    ungrouped = [n for n in class_map.names if n not in assigned]
    clash = sorted(set(groups) & set(ungrouped))
    if clash:
        raise ValueError(f"{what}: group name(s) {clash} equal an ungrouped class name")
    new_names = list(groups) + ungrouped
    new_index = {n: i for i, n in enumerate(new_names)}
    mapping = np.array([new_index[assigned.get(n, n)] for n in class_map.names], dtype=np.int64)
    return mapping, new_names


def merge_groups(labels: np.ndarray, class_map: ClassMap,
                 groups: Dict[str, List[str]]) -> Tuple[np.ndarray, ClassMap]:
    """
    Merge classes into groups before training

    Args:
        labels: (N,) labels in `class_map` numbering
        class_map: ClassMap (after removal)
        groups: {group name: [member class names]} (TOML order = label order)

    Returns:
        (new labels, new ClassMap): groups first, then the ungrouped classes in increasing
        old-label order with their names
    """
    mapping, new_names = _group_mapping(class_map, groups, "MergeGroups")
    return mapping[np.asarray(labels, dtype=np.int64)], ClassMap(new_names)


def posthoc_merge(probs: np.ndarray, class_map_trained: ClassMap,
                  groups: Dict[str, List[str]]) -> Tuple[np.ndarray, ClassMap, np.ndarray]:
    """
    Sum the probabilities of trained classes into groups (labels/plots only)

    Same semantics as BFM_WM `SNGP.merge_group_probabilities`, by name: groups first,
    ungrouped trained classes next, the untrained columns (K_tr..K-1) appended one by one.
    The confidence must always use the un-merged probabilities.

    Args:
        probs: (N, K) probabilities, the first K_tr columns are the trained classes
        class_map_trained: ClassMap of the K_tr trained classes
        groups: {group name: [trained class names]}

    Returns:
        (merged_probs (N, K'), merged ClassMap of the trained groups/classes,
         label_mapping (K,) int array old column -> new column)
    """
    probs = np.asarray(probs)
    k_tr, k = len(class_map_trained), probs.shape[1]
    if k < k_tr:
        raise ValueError(f"probs has {k} columns but {k_tr} trained classes")
    mapping_tr, new_names = _group_mapping(class_map_trained, groups or {}, "PosthocGroups")
    n_tr_new = len(new_names)
    label_mapping = np.concatenate([mapping_tr, n_tr_new + np.arange(k - k_tr, dtype=np.int64)])
    merged = np.zeros((probs.shape[0], n_tr_new + k - k_tr), dtype=np.float64)
    for old in range(k):
        merged[:, label_mapping[old]] += probs[:, old]
    return merged, ClassMap(new_names), label_mapping


def display_names(class_map: ClassMap, mapping: Dict[str, str]) -> List[str]:
    """
    Class names for display

    Args:
        class_map: ClassMap
        mapping: {class name: display name}; unknown keys only warn, since one table may
                 serve several models

    Returns:
        Display names in label order
    """
    mapping = mapping or {}
    unused = [k for k in mapping if k not in class_map.index]
    if unused:
        print(f"WARNING: DisplayNames entries not matching any class (ignored): {unused}")
    return [mapping.get(n, n) for n in class_map.names]


def class_counts(labels: np.ndarray, class_map: ClassMap) -> Dict[str, int]:
    """
    Number of points per class

    Args:
        labels: (N,) labels
        class_map: ClassMap of `labels`

    Returns:
        {class name: count} in label order (zeros included)
    """
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(class_map))
    return {n: int(counts[i]) for i, n in enumerate(class_map.names)}
