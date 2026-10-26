# ZEN-garden v4 domain/relation set migration handoff

## Instructions for the next chat

Continue this migration in the worktree `01_ZEN_garden_v4` on branch
`contribute--v4`. Before editing, verify the absolute worktree path, active branch,
worktree status, repository-local `AGENTS.md`, and current implementation. Do not
apply the changes to `01_ZEN_garden` or `02_ZEN-garden-plugins`.

Set the following as an active goal and pursue it until implementation and
verification are complete:

> Migrate ZEN-garden v4 to canonical `DomainSet` and `RelationSet` definitions
> with readable Cartesian indices, automatic/overridable index relations,
> derived structural masks, performance masks with explicit result defaults,
> removal of obsolete `BaseSet`/`SimpleSet`/`IndexedSet` abstractions, and
> preservation of ZEN-garden v3.0.0 result API behavior.

If work is interrupted or credits run out, leave the goal active and continue
from the authoritative worktree state when resumed.

## User intent

The aim is a simple and consistent modeling API in which writing sets,
parameters, variables, and constraints is clean and intuitive. The design must
distinguish combinations that are structurally meaningless from combinations
that are valid but omitted from the solver for performance.

Important compatibility requirements:

- Do not introduce breaking changes in result behavior.
- Preserve the outputs and signatures of key ZEN-garden v3.0.0 result methods,
  especially `get_total()` and `get_full_ts()`/full-time-series access.
- Structurally invalid combinations must not appear as valid, non-missing
  solution entries. Dense xarray outputs may retain their coordinates with NaN;
  sparse/tabular result views must omit them.
- Structurally valid entries removed only for performance must appear in saved
  results with their declared default, normally zero.
- Keep changes unstaged unless the user explicitly asks otherwise.

### Requirement precedence

This handoff incorporates the original migration plan plus later decisions from
the discussion. When they conflict, the later explicit user decision wins. The
only material override of the pasted original plan is the final treatment of
legacy set classes: the original allowed temporary compatibility views, while a
later instruction asked to remove `SimpleSet` and `IndexedSet` when no longer
necessary. Temporary adapters remain permissible during migration, but they are
not the desired completed architecture.

## Agreed architecture

### 1. Canonical data model

The following representation is not optional background: it is the concrete
design agreed in the discussion. In particular, `RelationSet` is backed by a
named pandas `MultiIndex`. It should not be implemented as a dense xarray mask,
a dictionary, or a plain list of tuples. Those forms may be accepted as input,
but registration normalizes them to a `MultiIndex`.

Use only two stored set types:

1. `DomainSet`: ordered, unique members of one named dimension.
2. `RelationSet`: allowed tuples across two or more named domain dimensions.

`GenericSet` may remain as a declarative class that knows how to obtain and
register set data; it is not a third stored set type.

#### Proposed `DomainSet`

```python
@dataclass(frozen=True, init=False)
class DomainSet:
    """An ordered, unique one-dimensional model domain."""

    name: str
    members: pd.Index
    doc: str = ""

    def __init__(
        self,
        name: str,
        members: pd.Index | Iterable[Any],
        doc: str = "",
    ) -> None:
        normalized = pd.Index(list(members))
        if normalized.has_duplicates:
            duplicates = normalized[normalized.duplicated()].unique().tolist()
            raise ValueError(
                f"Domain {name!r} contains duplicate members: {duplicates}"
            )
        normalized.name = name
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "members", normalized)
        object.__setattr__(self, "doc", doc)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.members)

    def __len__(self) -> int:
        return len(self.members)

    def __contains__(self, item: object) -> bool:
        return item in self.members
```

The canonical representation is `pd.Index`, not an xarray. The registry may
also expose it as an xarray coordinate. Input order is retained. Duplicate
members are rejected rather than silently removed or sorted.

#### Proposed `RelationSet`

```python
@dataclass(frozen=True, init=False)
class RelationSet:
    """Allowed combinations across two or more named model domains."""

    name: str
    dimensions: tuple[str, ...]
    combinations: pd.MultiIndex
    doc: str = ""
    apply_by_default: bool = True

    def __init__(
        self,
        name: str,
        dimensions: tuple[str, ...],
        combinations: (
            pd.MultiIndex
            | Iterable[tuple[Any, ...]]
            | Mapping[Any, Iterable[Any]]
        ),
        doc: str = "",
        apply_by_default: bool = True,
    ) -> None:
        if len(dimensions) < 2:
            raise ValueError("A relation requires at least two dimensions")
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("Relation dimensions must be unique")

        if isinstance(combinations, Mapping):
            if len(dimensions) != 2:
                raise ValueError(
                    "Mapping input is supported only for two-dimensional relations"
                )
            tuples = [
                (parent, child)
                for parent, children in combinations.items()
                for child in children
            ]
            index = (
                pd.MultiIndex.from_tuples(tuples, names=dimensions)
                if tuples
                else pd.MultiIndex.from_arrays(
                    [[] for _ in dimensions], names=dimensions
                )
            )
        elif isinstance(combinations, pd.MultiIndex):
            if combinations.nlevels != len(dimensions):
                raise ValueError("Relation arity does not match its dimensions")
            index = combinations.set_names(dimensions)
        else:
            tuples = list(combinations)
            if any(len(item) != len(dimensions) for item in tuples):
                raise ValueError("Relation contains tuples with invalid arity")
            index = (
                pd.MultiIndex.from_tuples(tuples, names=dimensions)
                if tuples
                else pd.MultiIndex.from_arrays(
                    [[] for _ in dimensions], names=dimensions
                )
            )

        if index.has_duplicates:
            duplicates = index[index.duplicated()].unique().tolist()
            raise ValueError(
                f"Relation {name!r} contains duplicate combinations: {duplicates}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "combinations", index)
        object.__setattr__(self, "doc", doc)
        object.__setattr__(self, "apply_by_default", apply_by_default)
```

Required convenience behavior:

```python
def select(self, **selectors: Any) -> pd.MultiIndex:
    """Return relation rows matching exact named-dimension selectors."""
    unknown = set(selectors).difference(self.dimensions)
    if unknown:
        raise KeyError(f"Unknown relation dimensions: {sorted(unknown)}")
    frame = self.combinations.to_frame(index=False)
    for dimension, value in selectors.items():
        frame = frame.loc[frame[dimension] == value]
    if frame.empty:
        return pd.MultiIndex.from_arrays(
            [[] for _ in self.dimensions], names=self.dimensions
        )
    return pd.MultiIndex.from_frame(frame, names=self.dimensions)

def project(self, dimension: str, **selectors: Any) -> pd.Index:
    """Select rows and return ordered unique members of one dimension."""
    if dimension not in self.dimensions:
        raise KeyError(f"Unknown relation dimension {dimension!r}")
    selected = self.select(**selectors).to_frame(index=False)
    return pd.Index(
        selected[dimension].drop_duplicates(),
        name=dimension,
    )
```

For example:

```python
technology_carrier = RelationSet(
    name="relation_technology_carrier",
    dimensions=("set_technologies", "set_carriers"),
    combinations={
        "boiler": ["gas", "heat"],
        "heat_pump": ["electricity", "heat"],
    },
)

technology_carrier.project(
    "set_carriers", set_technologies="boiler"
)
# pd.Index(["gas", "heat"], name="set_carriers")
```

The `MultiIndex` is the sparse structural source of truth. A dense xarray mask is
derived only when a backend operation needs one.

#### Registry storage

The registry should store the two concepts separately:

```python
self.sets: dict[str, DomainSet] = {}
self.relations: dict[str, RelationSet] = {}
self.coords_dataset = xr.Dataset()
```

The canonical registration API is explicit:

```python
add_domain(domain: DomainSet) -> None
add_relation(relation: RelationSet) -> None
```

Do not infer the canonical stored type merely because incoming data happens to
be a `Mapping`. A temporary legacy `add_set(..., index_set=...)` adapter may
translate old mapping-backed declarations into one domain plus one relation
during migration, but new and migrated code should register the concepts
explicitly and the adapter should be removed from the final state when possible.

`SetRegistry.__getitem__(name)` should return a `DomainSet`. Relation access is
through `relations[name]` or a dedicated method. Do not make `DomainSet` pretend
to support dictionary-style parent lookup. Replace old code such as:

```python
sets["set_input_carriers"][technology]
```

with a named relation projection:

```python
sets.get_related(
    "set_conversion_technologies",
    technology,
    "set_input_carriers",
)
```

The proposed helper returns a `pd.Index` and requires exactly one matching
two-dimensional relation. Zero or multiple matches should raise a clear error.

#### `ResolvedIndex` contract

```python
@dataclass(frozen=True)
class ResolvedIndex:
    dimensions: tuple[str, ...]
    combinations: pd.MultiIndex
    coordinates: dict[str, pd.Index]
    structural_mask: xr.DataArray
    applied_relations: tuple[str, ...]
```

Use a one-level `MultiIndex` even for a one-dimensional resolved index so the
resolver has one uniform contract. Callers that require a flat iterable must use
`combinations.get_level_values(0)` explicitly.

#### Exact resolver algorithm

Given readable domain names and `index_relations`:

1. Validate that every requested domain exists. Register documented derived
   domains and relations in a dedicated set-construction phase before any
   component is resolved; `resolve()` should be read-only and should not mutate
   the registry as a side effect.
2. For `"auto"`, select all relations with `apply_by_default=True` whose complete
   dimension set is a subset of the requested dimensions. Reject ambiguous
   defaults as described below before joining them.
3. For an explicit tuple, validate every name and select exactly those relations.
4. For `()`, select no relations.
5. Convert each selected relation's `MultiIndex` with
   `relation.combinations.to_frame(index=False)`.
6. Natural-join relation frames when they share dimension columns:

   ```python
   frame = frame.merge(relation_frame, on=shared, how="inner", sort=False)
   ```

7. Cross-join relation frames that share no dimensions:

   ```python
   frame = frame.merge(relation_frame, how="cross", sort=False)
   ```

8. If no relations were selected, initialize the join with a one-row,
   zero-column frame: `pd.DataFrame(index=[0])`. This is the neutral element for
   the following Cartesian joins.
9. For every requested dimension absent from the joined frame, cross-join the
   corresponding `DomainSet.members`.
10. Reorder columns to the exact declared `indices` order without sorting.
    Validate that the resolved rows are unique; raise a clear error if a join
    unexpectedly produced duplicate combinations instead of silently dropping
    them.
11. Convert the frame to a named `pd.MultiIndex`.
12. Build canonical coordinates from the complete `DomainSet.members`, then
    create a boolean xarray structural mask initialized to `False` and assign
    `True` at the allowed `MultiIndex` tuples.

For `"auto"`, consume relations in registry insertion order. For an explicit
tuple, consume them in the tuple's order. Although the allowed set should be
join-order independent, deterministic ordering matters for reproducible output
indices; test it rather than relying on undocumented pandas sorting behavior.

For automatic selection, more than one default relation with the same normalized
dimension signature is ambiguous and must raise a clear error rather than being
silently intersected or resolved by registration order. Relations with different
dimension signatures may all apply and are naturally joined. If two
same-signature relations genuinely need to be intersected, mark at least one
non-default and select both explicitly through `index_relations=(...)`.

Illustrative core:

```python
dimensions = tuple(indices)
coordinates = {
    dimension: self.sets[dimension].members
    for dimension in dimensions
}
structural_mask = xr.DataArray(
    False,
    coords=coordinates,
    dims=dimensions,
)
if len(combinations):
    selectors = tuple(
        xr.DataArray(
            combinations.get_level_values(dimension),
            dims="_entry",
        )
        for dimension in dimensions
    )
    structural_mask.loc[selectors] = True
```

Use vectorized selectors sharing one temporary selector dimension so tuple rows
are selected pairwise rather than as a Cartesian product. Give every selector
the same explicit temporary dimension, for example `"_entry"`; do not rely on
xarray assigning matching anonymous dimension names:

```python
selectors = tuple(
    xr.DataArray(
        combinations.get_level_values(dimension),
        dims="_entry",
    )
    for dimension in dimensions
)
```

Relations whose dimensions are only partially present must not be applied. For
example, a relation over `(technology, location)` cannot filter a component that
declares only `technology`.

Dimension-name matching is exact. Do not silently treat
`set_conversion_technologies` as `set_technologies` during relation selection,
even though its members form a subset. Register a relation against the exact
domain used by a component, or make the component use the common parent domain
when that is semantically correct. Mask projection may have explicit,
well-validated aliases, but relation resolution must not guess them.

#### Relation registration validation

Normalize the relation first, then validate:

```python
missing_dimensions = set(dimensions).difference(self.sets)
if missing_dimensions:
    raise KeyError(...)

for dimension in dimensions:
    invalid = relation.combinations.get_level_values(dimension).difference(
        self.sets[dimension].members
    )
    if len(invalid):
        raise ValueError(...)
```

When migrating an old dictionary-backed indexed set, registration should:

1. flatten its child values and explicitly derive ordered unique domain members,
   for example with `pd.Index(flattened).drop_duplicates()`, before constructing
   the child `DomainSet`;
2. register that child domain;
3. register the parent/child `RelationSet` from the original mapping;
4. name it deterministically, for example
   `relation_conversion_technologies_input_carriers` after stripping `set_`.

Register the child domain before the relation so validation succeeds.

Track legacy serialization separately from the semantic relation object, for
example:

```python
self.legacy_indexed_set_relations: dict[str, str] = {
    child_domain_name: relation_name,
}
```

Populate this mapping only while migrating an old dictionary-backed indexed set.
Do not infer persistence from a relation's name, and do not serialize every
derived relation into the legacy `sets.h5`; that would change established output
content. New relation metadata may be written to a separate private file if it
is useful, but the v3-compatible loader must not depend on it.

### 2. Type removal and backend boundaries

Remove `BaseSet`, `SimpleSet`, and `IndexedSet` if they are no longer required.
Do not retain compatibility wrappers merely to preserve the old internal
hierarchy.

The original migration plan allowed temporary `SimpleSet`/`IndexedSet`
compatibility views while built-in elements and plugins migrated. A later user
instruction tightened the final state: remove those classes if they are no
longer necessary. Therefore adapters are permitted only as a short-lived
implementation stage when concrete unmigrated callers require them; they are not
an accepted final deliverable. Prefer migrating all in-scope callers and deleting
the adapters. If an adapter must remain because an external plugin demonstrably
depends on it, document that as an unresolved compatibility limitation rather
than calling the goal fully complete.

The representation boundary is intentional:

- pandas `Index`/`MultiIndex` is canonical for domain membership, relation
  tuples, projections, joins, and sparse structural resolution;
- xarray is canonical for named dense coordinates, parameters, masks, and
  aligned model expressions;
- Linopy receives xarray coordinates and masks.

Do not force all sets to be xarrays. That would make sparse relations and
projections less clear. Do not expose incidental pandas operations through the
public Results API.

### 3. Readable Cartesian index declarations

Keep component index declarations readable:

```python
indices = (
    "set_technologies",
    "set_location",
    "set_capacity_types",
    "set_years",
)
```

Do not require composite names such as `set_technology_location` or
`set_technology_capacity_type` in variable and parameter declarations. Use the
actual domain names defined by the v4 repository consistently; the examples in
this handoff use the existing plural technology/capacity-type conventions.

### 4. Index relations

Components may declare:

```python
index_relations = "auto"  # default
```

Semantics:

- `"auto"`: apply every default relation whose dimensions are contained in the
  component's declared domains.
- An explicit tuple of relation names: apply exactly those relations.
- `()`: request the unfiltered Cartesian product.

Relations should be natural-joined on shared dimensions and cross-joined when
independent. The resolver should return, at minimum:

- ordered dimension names;
- structurally allowed combinations;
- canonical coordinates;
- a dense xarray structural mask for Linopy;
- the names of applied relations.

Registration must validate that every relation dimension exists and every tuple
member belongs to the corresponding domain. Preserve deterministic domain and
tuple order. Applicable default relations with distinct dimension signatures
are all applied and intersect through natural joins. Ambiguous defaults with the
same normalized dimension signature are rejected. Relations that should not
participate automatically must set `apply_by_default=False`; callers can opt
into them explicitly.

Do not support a direct `structural_mask_override`. Exceptional components must
choose relations through `index_relations`; otherwise structural validity would
have two competing sources of truth.

### 5. Structural examples

`set_location` should contain the union of nodes and edges as its domain, while
the technology/location relation contains only meaningful pairs:

- conversion, storage, and retrofitting technologies at nodes;
- transport technologies on edges.

Likewise, a technology/capacity-type relation should contain only meaningful
technology and capacity-type pairs. In the standard model, storage technologies
may use both power and energy capacity, while non-storage technologies must not
receive energy-capacity entries unless the domain model explicitly permits it.

The domain may therefore be broad and reusable; the relation defines allowed
combinations. A component declaring the readable Cartesian domains receives the
relation-derived structural mask automatically.

Audit and register explicit relations for every genuinely structural dependency,
including at least:

- technology/location validity;
- technology/capacity-type validity;
- technology/carrier roles;
- technology/existing-vintage validity;
- edge/node connectivity;
- retrofit/base-technology dependencies;
- on/off eligibility, but only if ineligible tuples truly have no model meaning
  rather than merely being removable for performance.

Do not mechanically convert every existing mask into a relation. Whether a
condition is structural is determined by result semantics: if a zero/default row
must remain meaningful in results, it is a performance mask instead.

### 6. Structural versus performance masking

The conceptual rule is:

```python
solver_mask = structural_mask & performance_mask
```

- `structural_mask` is derived automatically from domain relations. False means
  the combination has no model meaning. It remains NaN in dense arrays and is
  omitted from sparse/tabular result views.
- `performance_mask` is optional and excludes valid entries from the solver.
  False means the entry remains part of the result domain and must be restored
  to a declared default value.

When no performance mask is supplied, treat it as an all-`True` mask, so
`solver_mask == structural_mask`.

A performance mask may remove only structurally valid entries. It can never
reactivate a structurally invalid combination because the solver mask is always
the logical intersection.

The required behavior is:

| Structural | Performance | Solver entry | Dense result | Sparse result |
|---|---|---|---|---|
| `True` | `True` | Present | Solver value | Solver value |
| `True` | `False` | Omitted | `mask_default_value` | Default-valued row |
| `False` | Either | Omitted | NaN | Row omitted |

The low-level variable API should expose fields equivalent to:

```python
mask = performance_mask
mask_default_value = 0.0
```

The user explicitly preferred `mask` and `mask_default_value` over an
`activation` field. At the declarative level, reusable masks should be described
as performance masks, not structural masks.

For this migration, zero is the only supported solver-elimination default.
Reject nonzero `mask_default_value` values explicitly: absent Linopy variables
behave as zero in expressions, so restoring a nonzero result afterward would not
match the solved model. Never silently ignore a supplied default.

Saved variable results must restore the default only where:

```python
structural_mask & ~performance_mask
```

Structural exclusions must remain missing. This distinction must survive
postprocessing and the public Results API, not just exist as registry metadata.

### 7. Exact component API

The preferred declarative base fields were:

```python
class GenericVariable:
    indices: ClassVar[list[str] | tuple[str, ...]]
    index_relations: ClassVar[Literal["auto"] | tuple[str, ...]] = "auto"
    masks: ClassVar[list[type[GenericMask]]] = []
    mask_default_value: ClassVar[float | None] = None
```

The word `activation` was explicitly rejected. The public/declarative names are:

- `index_relations` for structural relation selection;
- `masks` (or low-level `mask`) for optional performance masks;
- `mask_default_value` for result restoration.

Use `None` as the base default. Require a concrete value whenever a performance
mask is supplied, and reject a default when no performance mask exists. Existing
performance-masked variables should declare `mask_default_value = 0.0`
explicitly. This prevents an accidental assumption that every future mask means
zero.

#### Worked variable example

```python
class Capacity(GenericVariable):
    name = "capacity"
    indices = (
        "set_technologies",
        "set_capacity_types",
        "set_location",
        "set_years",
    )
    index_relations = "auto"
    masks = [CapacityAvailableMask]
    mask_default_value = 0.0
```

Assume:

```text
set_technologies = [boiler, battery, pipeline]
set_capacity_types = [power, energy]
set_location = [node_a, node_b, edge_ab]
```

Relations allow:

```text
(boiler, power)
(battery, power)
(battery, energy)
(pipeline, power)

(boiler, node_a), (boiler, node_b)
(battery, node_a), (battery, node_b)
(pipeline, edge_ab)
```

The resolver natural-joins those relations on `set_technologies`, then
cross-joins years. Thus `(pipeline, power, node_a, 2030)` is structural false and
must never become a meaningful result row. If `CapacityAvailableMask` is false
for `(boiler, power, node_b, 2030)`, that tuple is structurally valid but omitted
from the solver and restored to zero in results.

#### Build sequence for variables

```python
resolved = model_constructor.resolve_indices(cls.indices, cls.index_relations)

index_values = (
    list(resolved.combinations.get_level_values(0))
    if len(resolved.dimensions) == 1
    else list(resolved.combinations)
)
index_sets = (index_values, list(resolved.dimensions))

performance_mask = masks.combine_for_indices(
    cls.masks,
    model_constructor,
    index_sets,
)

variables.add_variable(
    name=cls.name,
    index_sets=index_sets,
    bounds=cls.get_bounds(model_constructor, index_sets),
    mask=performance_mask,
    structural_mask=resolved.structural_mask,
    mask_default_value=(
        cls.mask_default_value if performance_mask is not None else None
    ),
    applied_relations=resolved.applied_relations,
)
```

At the registry/backend boundary, use the relation-derived structural mask as
the canonical solver index mask and intersect the optional performance mask:

```python
solver_mask = structural_mask
if performance_mask is not None:
    solver_mask = solver_mask & performance_mask
```

Store all of the following separately for diagnostics and postprocessing:

```python
structural_masks: dict[str, xr.DataArray]
performance_masks: dict[str, xr.DataArray]
mask_default_values: dict[str, float]
applied_relations: dict[str, tuple[str, ...]]
```

Do not overwrite `structural_mask` with the combined solver mask.

#### Worked parameter example

```python
class ConversionFactor(GenericParameter):
    name = "conversion_factor"
    indices = (
        "set_conversion_technologies",
        "set_dependent_carriers",
        "set_nodes",
        "set_years",
        "set_hours",
    )
    index_relations = "auto"
```

The technology/dependent-carrier relation filters meaningless carrier pairs.
Nodes, years, and hours remain Cartesian unless their own complete relations are
declared. Parameter input is aligned only to the resolved combinations. Missing
structural tuples are not manufactured as ordinary parameter values. If supplied
parameter data contains a tuple outside the resolved structural combinations,
raise a clear validation error; do not silently drop it or expand the relation.

#### Worked reusable performance mask

```python
class PositiveCapacityEmissionMask(GenericMask):
    name = "positive_capacity_emission_mask"

    @classmethod
    def build(cls, model_constructor, registry) -> xr.DataArray:
        capacity = model_constructor.optimization_model.parameters.capacity_existing
        return (capacity > 0).astype(bool)
```

This is not a relation because a zero capacity does not make the
technology/location/year combination meaningless. It is an optimization choice:
emission variables or constraints can be omitted there, while the result should
still report emissions as zero.

#### Worked constraint example

With structural relations already reflected in variable support, constraint code
can stay domain-readable:

```python
flow = variables["flow_conversion_input"]
factor = parameters.conversion_factor
reference_flow = flow.where(reference_carrier_mask).sum("set_input_carriers")

lhs = flow.rename(
    {"set_input_carriers": "set_dependent_carriers"}
) - factor * reference_flow

constraints.add_constraint(
    "constraint_carrier_conversion",
    lhs == 0,
    mask=power_operation_mask,  # performance mask only
)
```

The relation controls which technology/carrier variable entries exist. The
constraint author should not have to index a dictionary-backed set by technology
or manually build the structural Cartesian mask. A relation projection may still
be used to construct semantic selections such as the reference carrier mask.

#### Exact result restoration rule

Linopy represents both structural and performance-masked variables as missing in
its dense solution arrays, so postprocessing must distinguish them using stored
metadata:

```python
result = xr.where(
    ~structural_mask,
    np.nan,
    xr.where(
        performance_mask,
        solver_solution,
        mask_default_value,
    ),
)
```

The implementation may use `.where()` as below, but it must be semantically
equivalent to this formula.

```python
def restore_masked_defaults(solution: xr.Dataset) -> xr.Dataset:
    restored = solution.copy()
    for name, performance_mask in performance_masks.items():
        values = restored[name]
        structural = align_mask(structural_masks[name], values)
        performance = align_mask(performance_mask, values)
        use_default = structural & ~performance
        restored[name] = values.where(
            ~use_default,
            mask_default_values[name],
        )
    return restored
```

Here `align_mask()` must validate and handle the empty-dimension renaming case
described below before calling `reindex_like(..., fill_value=False)`. Do not use
plain `reindex_like` as the only alignment step when Linopy has renamed an empty
dimension.

Write this restored dataset to `variables.nc`. Do not call `fillna(0)` globally,
because that would turn structurally impossible combinations into apparent zero
solutions.

Do not mutate `lp_model.solution` in place. Solver diagnostics and later logic
may need the raw Linopy solution. Restore defaults in a copy used for persistence
and public result loading.

For sparse pandas reconstruction, the equivalent rule is to reindex onto
`resolved.combinations` and fill only those missing rows. Since
`resolved.combinations` contains only structural tuples, structurally invalid
tuples are never introduced.

#### Empty-domain/xarray/Linopy edge case

Linopy may rename an empty dimension with a UUID when the model already contains
a coordinate with the same name. Therefore a stored structural or performance
mask can have semantically identical dimensions but names different from the
mask produced by the legacy bounds/index helper.

When shapes and positional dimensions correspond, rename masks positionally to
the backend's actual dimensions before `reindex_like`. Do this only at the
backend boundary and validate dimension count/order; do not allow arbitrary
silent renaming.

Also construct xarrays with explicit dimensions:

```python
coords = {
    name: sets.get_coord(data, name).data
    for data, name in zip(index_arrays, index_names, strict=True)
}
lower = xr.DataArray(0.0, coords=coords, dims=index_names)
```

Avoid `xr.DataArray(..., coords=[...])` when coordinate objects may retain old
dimension metadata, because xarray can infer an unintended dimension name.

### 8. Constraint-specific rules

Constraints should be written against aligned xarray/Linopy expressions over
the readable domains. Structural validity should come from resolved relations
and variable/expression support; optional reusable masks are performance masks.
Do not require constraint authors to repeatedly split composite sets or manually
reconstruct technology/location membership.

Handle constant-only constraint rows explicitly: omit rows that are already
satisfied and raise a clear error for violated constant rows. Do not hide invalid
rows by indiscriminately filling NaNs.

For an expression with a Linopy `_term` dimension, a constraint row has variable
support only where at least one variable label is nonnegative. Every coefficient
belonging to an active variable term must be finite; mask out a row containing a
non-finite active coefficient so it is never sent to Linopy. Combine variable support, finite
active coefficients, finite RHS/sign data, and any declared performance mask.
For rows with no variable support, evaluate the constant relation before deciding
whether to omit or reject it. Confirm the exact Linopy expression fields against
the version used by v4 rather than assuming an API from another version.

### 9. Backend and memory expectations

Relations improve semantics and sparse tuple construction, but dense xarray
coordinates/masks can still have Cartesian memory size. Do not claim that the
design inherently eliminates dense backend memory use. Use sparse allowed
combinations where Linopy accepts them, while retaining dense aligned masks only
where required by xarray/Linopy.

### 10. Time-index context from the earlier discussion

The broader refactor also sought to separate years and hours instead of using a
single long `set_time_steps_operation` list. Preserve these naming decisions if
they already exist on `contribute--v4`:

- `weight_representative_hours` (singular `weight`, not `weights`);
- `duration_representative_hours_storage`.

The number of representative hours is an invariant across modeled years,
including year-specific time-series aggregation. The profiles and weights may
differ by year, but every year uses the same representative-hour cardinality.
Therefore `set_representative_hours` should be one shared domain, not a
dictionary indexed by year merely to repeat the same labels. Year specificity
belongs in parameters indexed by `(set_years, set_representative_hours)` and, if
needed, explicit mappings from original to representative hours. Do not create a
year/representative-hour relation when every year permits every representative
hour; that relation would only restate the Cartesian product. Inspect the v4
implementation before changing this area, because it may already contain the
earlier time-index migration.

Operational components should therefore declare year and hour dimensions
separately, for example:

```python
indices = (
    "set_conversion_technologies",
    "set_input_carriers",
    "set_nodes",
    "set_years",
    "set_representative_hours",
)
```

Do not rebuild a flattened domain containing `(year, hour)` values merely to
recreate `set_time_steps_operation`. Parameters that vary by year and hour,
including representative-hour weights, carry both dimensions explicitly.
Constraint code can then sum or select `set_years` and
`set_representative_hours` directly without repeatedly splitting a compound time
index. Legacy flattened time-step structures may still be reconstructed inside
the Results compatibility layer where v3.0.0 output behavior requires them.

## Result compatibility requirements

Use the actual v3.0.0 implementation as the compatibility reference, not memory
or assumptions. Verify signatures and output structure for at least:

- `Results.get_total(...)`;
- `Results.get_full_ts(...)` or the v4 equivalent retaining that public behavior;
- saved `variables.nc` dimensions, structural NaNs, and performance-mask zeros;
- indexed-set/set-relation serialization consumed by result loading;
- rolling-horizon state transfer for performance-masked variables.

The signatures inspected during the earlier work were:

```python
Results.get_total(
    component_name,
    year=None,
    scenario_name=None,
    keep_raw=False,
    index=None,
)

Results.get_full_ts(
    component_name,
    scenario_name=None,
    discount_to_first_step=True,
    year=None,
    keep_raw=False,
    index=None,
)
```

Re-check these against the v3.0.0 tag in the target repository, but do not rename
`get_full_ts` merely because prose sometimes called it `get_full_time_series`.

Do not make relation metadata or new internal dimensions leak into established
public outputs unless the old API already exposed an equivalent structure.
Internal renames such as `weight_representative_hours` and
`duration_representative_hours_storage` must likewise use compatibility aliases
at serialization/loading boundaries if v3.0.0 files expose older key names. The
internal cleanup does not authorize changing established file schemas.

## Why relations were retained instead of structural masks alone

This was explicitly discussed. A standalone structural mask can filter a
particular component, but it does not provide a reusable semantic source of
truth for questions such as:

- Which locations belong to a technology?
- Which carriers are inputs of this conversion technology?
- Which capacity types are meaningful for this technology?
- Which tuples should a newly declared parameter inherit automatically?

A `RelationSet` answers these questions sparsely, supports projections and joins,
and generates structural masks consistently for every component. Without it,
each component would need to recreate or name its own mask, producing duplicated
logic and making constraint code less readable.

The agreed split was therefore:

- domain sets define valid individual labels;
- relation sets define valid combinations and are the structural source of truth;
- structural masks are derived backend artifacts;
- performance masks express optional solver reductions.

Do not model relations only as dense boolean xarrays. That loses the sparse tuple
API and makes semantic projection awkward. Conversely, do not expect the
`MultiIndex` alone to save all backend memory: xarray/Linopy may still require a
dense Cartesian mask.

## Required diagnostics

Expose per-component diagnostics with at least:

- requested dimensions;
- applied index relations;
- dense Cartesian cell count;
- structurally valid count;
- performance-mask active count after alignment to the dense component index;
- final solver entry count;
- dense/active ratio.

The counts must follow the mask semantics rather than summing independently
misaligned arrays. In particular:

```python
structurally_valid = structural_mask.sum()
performance_active = performance_mask.sum()
solver_entries = (structural_mask & performance_mask).sum()
dense_active_ratio = dense_cells / solver_entries
```

Treat an all-`True` performance mask as implicit when none is declared. Define
the zero-entry ratio explicitly (for example infinity when `dense_cells > 0`)
instead of dividing silently. Warn when the dense/active ratio is at least 5.
These diagnostics are part of the migration acceptance criteria, not optional
debug logging.

## Suggested implementation sequence

1. Inspect `01_ZEN_garden_v4`, confirm `contribute--v4`, local instructions,
   worktree cleanliness, and current time-index/set work.
2. Inventory all set classes, indexed-set reads, custom Cartesian-set helpers,
   variable/parameter index declarations, reusable masks, constraint masking,
   result reconstruction, and set serialization.
3. Add canonical, strongly typed `DomainSet`, `RelationSet`, and a resolved-index
   value object.
4. Refactor the set registry to register domains/relations, validate relations,
   resolve readable Cartesian declarations, and expose clear relation queries.
5. Migrate declarative parameters and variables to `indices` plus
   `index_relations`.
6. Derive and register technology/location and technology/capacity-type
   relations so only allowed entries become structurally present.
7. Replace direct dictionary-style indexed-set reads with relation projections
   or aligned structural masks.
8. Separate structural masks from performance masks throughout variable and
   constraint construction; support `mask_default_value`.
9. Restore performance-masked defaults during result persistence while keeping
   structural exclusions absent.
10. Add the required per-component structural/performance density diagnostics
    and the ratio warning threshold.
11. Preserve legacy result/set serialization where the v3 loader depends on it.
12. Remove obsolete set classes and dead custom-set/indexing-set code only after
    all call sites have migrated.
13. Add or update focused tests only if permitted by the repository/user
    instructions, then run formatting, linting, typing, unit tests, representative
    end-to-end cases, and explicit Results API comparisons.

## Work previously attempted in the wrong repository

An implementation was partially developed in `01_ZEN_garden` on its `main`
branch before the user clarified that the correct target is the v4 worktree.
Those edits are unstaged and must not be treated as changes to deliver. They may
be inspected only if both repositories are workspace-accessible and doing so is
useful as a design reference; re-implement against the authoritative v4 code
rather than blindly copying patches.

That wrong-repository attempt established useful evidence:

- canonical domain/relation objects and relation-based resolution were viable;
- baseline end-to-end case `test_1a` passed after migration fixes;
- transport/storage cases exposed special handling needed for empty dimensions,
  because Linopy can rename empty coordinate dimensions;
- explicit xarray `dims` are safer than relying on dimension inference from
  coordinate objects;
- inherited input/output variable implementations must query the relation using
  the actual declared carrier dimension rather than hard-coding input carriers;
- result persistence must restore performance defaults independently of
  structural exclusions;
- serializing every derived relation would change legacy output, so only
  relations corresponding to legacy declared indexed sets should be emitted in
  compatibility files unless v4 has a new private metadata format.

Do not assume paths or line numbers from that attempt match v4.

## Verification checklist

Before marking the goal complete, gather direct evidence for every item:

- The active branch is `contribute--v4` in `01_ZEN_garden_v4`.
- No `BaseSet`, `SimpleSet`, or `IndexedSet` implementation or live reference
  remains unless a documented, demonstrated requirement makes one necessary.
- Domain registration preserves order and uniqueness.
- Domain and relation registration reject duplicate members/combinations.
- Relation registration rejects invalid arity, unknown dimensions, and
  out-of-domain members.
- Automatic, explicit, and disabled relation resolution all work.
- Relation `select()`/`project()`, natural joins on shared dimensions,
  independent Cartesian crosses, explicit overrides, and Cartesian opt-out are
  manually verified if existing tests do not cover them.
- Ambiguous default relations fail clearly rather than being chosen silently.
- Technology/location structural combinations contain nodes for node-based
  technologies and edges for transport technologies only.
- Storage technologies alone receive energy-capacity entries under the standard
  capacity-type rules.
- Variable and parameter declarations remain readable Cartesian domain lists.
- Performance masks use explicit defaults; unsupported defaults raise errors.
- Linopy labels exist exactly where
  `structural_mask & performance_mask` is `True`.
- Saved results restore valid performance-masked entries to zero/default.
- Structurally invalid combinations remain NaN in dense results and omitted from
  sparse/tabular public results.
- Constraint construction remains clean and representative constraints solve.
- v3.0.0-compatible `get_total()` and full-time-series calls have unchanged
  signatures and equivalent outputs on representative fixtures.
- Representative before/after scenarios match in objective value, expected
  solver row/column counts, serialized dimension names/shapes, `get_total()`,
  and `get_full_ts()`.
- Per-component diagnostics contain all required counts and warn at a
  dense/active ratio of at least 5.
- A sparse synthetic technology/location case records dense and active counts;
  no Python-memory saving is claimed merely because relations are sparse.
- Representative conversion, storage, transport, empty-set, and rolling-horizon
  end-to-end cases pass.
- Ruff passes on changed files.
- Black check passes on changed files.
- mypy passes on changed files, or unrelated baseline failures are identified
  precisely.
- Relevant unit and end-to-end tests actually ran and passed.
- `git diff` contains no unrelated edits, and all changes remain unstaged.

## Repository safety and working rules

- Follow the target repository's `AGENTS.md` before taking task actions.
- Inspect relevant code and tests first.
- Prefer the smallest coherent, maintainable implementation.
- Do not add dependencies without justification.
- Do not modify unrelated code.
- Do not suppress errors or disable tests.
- Do not create or modify tests, fixtures, or test cases unless the user has
  explicitly authorized that in the target chat. Existing tests may and should
  be run.
- Do not claim a check passed unless it ran.
- State any remaining weaknesses or compatibility risks honestly.
- Do not revert or delete the wrong-repository unstaged changes unless the user
  explicitly requests cleanup.

## Prior goal status

The previous chat marked its goal `blocked` only because `01_ZEN_garden_v4` was
not available as a VS Code workspace root. Once that worktree is accessible,
start/resume the goal above and proceed from the v4 worktree's authoritative
state.
