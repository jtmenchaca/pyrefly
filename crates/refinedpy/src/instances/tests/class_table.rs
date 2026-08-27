use super::*;

// --- class_table: field order + declared sets ---

#[test]
fn class_table_reads_fields_in_declaration_order_with_declared_sets() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field, BaseModel\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Person(BaseModel):\n",
        "    age: Age\n",
        "    label: str\n",
    ));
    let aliases = crate::surface::compile_aliases(&module);
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let person = table.get("Person").expect("Person class recorded");
    assert_eq!(person.fields.len(), 2);
    assert_eq!(person.fields[0].name, "age");
    assert!(person.fields[0].declared.is_some(), "age reads its Annotated set");
    assert_eq!(person.fields[1].name, "label");
    assert!(person.fields[1].declared.is_none(), "bare str states no refinement");
}

/// A12.xfer.typeops's own real seeding path: `class_table` — not
/// `typed_dict_table` — is what `check.rs::seed_parameters` reads for
/// a bare `r: Record` PARAMETER (`context.classes` is checked before
/// any TypedDict-specific fallback), so a `Required[Age]` member must
/// read its own refinement HERE too, not only through
/// `typed_dict_table`'s separate return-position table. Before
/// `unwrap_required_marker` reached this loop, `Required`'s wrapper
/// made `declared_refinement` decline the field entirely, so `a`
/// never reached `class_parameter_object`'s seeded `keys`, and a
/// later `r["a"]` read wrongly proved `KeyError`.
#[test]
fn class_table_reads_a_typed_dicts_required_wrapped_field() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated, Required, TypedDict\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Record(TypedDict, total=False):\n",
        "    a: Required[Age]\n",
    ));
    let aliases = crate::surface::compile_aliases(&module);
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let record = table.get("Record").expect("Record class recorded");
    assert_eq!(record.fields.len(), 1);
    assert_eq!(record.fields[0].name, "a");
    assert!(record.fields[0].declared.is_some(), "Required[Age] must still read Age's own refinement");
}

// --- typed_dict_table: per-member refinements ---

#[test]
fn typed_dict_table_reads_each_members_own_refinement() {
    let module = parsed(concat!(
        "from typing import Annotated, TypedDict\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class PersonDict(TypedDict):\n",
        "    age: Age\n",
        "    label: str\n",
    ));
    let aliases = crate::surface::compile_aliases(&module);
    let imports = crate::surface::surface_imports(&module);
    let table = typed_dict_table(&module, &aliases, &imports);
    let members = table.get("PersonDict").expect("PersonDict recorded");
    assert_eq!(
        members.len(),
        2,
        "age reads its alias's refinement; bare str reads its base sort"
    );
    assert_eq!(members[0].name, "age");
    assert_eq!(members[0].declared.spelling, "Age");
    assert_eq!(members[1].name, "label");
    assert_eq!(
        members[1].declared.spelling, "str",
        "a plain builtin member falls back to its base sort, the same source a class field's own base_sort reads"
    );
    for member in members {
        assert!(
            member.required,
            "no `total=` keyword makes every member required (library/typing.rst, TypedDict)"
        );
    }
}

/// A bare builtin member (`a: int`) reaches the table through its BASE
/// SORT — the exact shape A8.sink.assign/A8.sink.ret declare. Without
/// the fallback `declared_refinement` declines a bare `int` (it is not
/// an alias), the member list comes back EMPTY, and the MEMBERS LAW
/// iterates zero times and answers Silent, so no member fire and no
/// missing-required-key fire is reachable for such a class at all.
#[test]
fn typed_dict_table_reads_bare_builtin_members_through_their_base_sort() {
    let module = parsed(concat!(
        "from typing import TypedDict\n",
        "class P(TypedDict):\n",
        "    a: int\n",
        "    b: int\n",
    ));
    let aliases = crate::surface::compile_aliases(&module);
    let imports = crate::surface::surface_imports(&module);
    let table = typed_dict_table(&module, &aliases, &imports);
    let members = table.get("P").expect("P recorded");
    assert_eq!(members.len(), 2, "both bare-int members reach the table");
    assert_eq!(members[0].name, "a");
    assert_eq!(members[1].name, "b");
    for member in members {
        assert!(member.required, "no `total=` keyword makes both members required");
    }
}

/// A12.xfer.typeops's own shape: `Required[Age]` on a `total=False`
/// class must record the SAME member `age: Age` (unwrapped) would —
/// `Required`/`NotRequired` state only presence, never a different
/// key set. Before `unwrap_required_marker`, `declared_refinement`
/// declined the whole `Subscript` (an unrecognized head), so the
/// member never reached this table at all, leaving the seeded
/// parameter's own `keys` without `"a"` and a later `r["a"]` read
/// wrongly proving `KeyError`.
#[test]
fn typed_dict_table_reads_a_required_wrapped_members_own_refinement() {
    let module = parsed(concat!(
        "from typing import Annotated, Required, TypedDict\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Record(TypedDict, total=False):\n",
        "    a: Required[Age]\n",
    ));
    let aliases = crate::surface::compile_aliases(&module);
    let imports = crate::surface::surface_imports(&module);
    let table = typed_dict_table(&module, &aliases, &imports);
    let members = table.get("Record").expect("Record recorded");
    assert_eq!(members.len(), 1, "Required[Age] must still read Age's own refinement");
    assert_eq!(members[0].name, "a");
    assert_eq!(members[0].declared.spelling, "Age", "Required peels to the wrapped annotation's own spelling");
    assert!(
        members[0].required,
        "Required overrides the class's own total=False for this one key"
    );
}

/// The `NotRequired[X]` twin, on an otherwise `total=True` class —
/// same peel, same member table shape.
#[test]
fn typed_dict_table_reads_a_not_required_wrapped_members_own_refinement() {
    let module = parsed(concat!(
        "from typing import Annotated, NotRequired, TypedDict\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Record(TypedDict):\n",
        "    a: NotRequired[Age]\n",
    ));
    let aliases = crate::surface::compile_aliases(&module);
    let imports = crate::surface::surface_imports(&module);
    let table = typed_dict_table(&module, &aliases, &imports);
    let members = table.get("Record").expect("Record recorded");
    assert_eq!(members.len(), 1, "NotRequired[Age] must still read Age's own refinement");
    assert_eq!(members[0].name, "a");
    assert_eq!(members[0].declared.spelling, "Age");
    assert!(
        !members[0].required,
        "NotRequired overrides the class's own total=True default for this one key"
    );
}

/// `total=False` with no per-key marker makes every member NOT required
/// — library/typing.rst, `TypedDict`: "It is also possible to mark all
/// keys as non-required by default by specifying a totality of
/// ``False``... a ``Point2D`` ``TypedDict`` can have any of the keys
/// omitted."
#[test]
fn typed_dict_table_reads_a_total_false_classs_members_as_not_required() {
    let module = parsed(concat!(
        "from typing import Annotated, TypedDict\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Record(TypedDict, total=False):\n",
        "    a: Age\n",
    ));
    let aliases = crate::surface::compile_aliases(&module);
    let imports = crate::surface::surface_imports(&module);
    let table = typed_dict_table(&module, &aliases, &imports);
    let members = table.get("Record").expect("Record recorded");
    assert_eq!(members.len(), 1);
    assert!(!members[0].required, "total=False makes every member not required");
}

#[test]
fn typed_dict_table_ignores_a_class_with_no_typed_dict_base() {
    let module = parsed(concat!(
        "from pydantic import BaseModel\n",
        "class Person(BaseModel):\n",
        "    age: int\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = typed_dict_table(&module, &aliases, &imports);
    assert!(table.get("Person").is_none(), "a plain BaseModel class is not a TypedDict");
}

// --- ClassVar is skipped ---

#[test]
fn class_var_annotated_row_is_not_an_instance_field() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import ClassVar\n",
        "class Counted:\n",
        "    total: ClassVar[int] = 0\n",
        "    age: int = 40\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let counted = table.get("Counted").expect("Counted class recorded");
    assert_eq!(counted.fields.len(), 1, "ClassVar row must not become a field");
    assert_eq!(counted.fields[0].name, "age");
}

// --- a field annotated with another module-level BaseModel class ---

/// m-pydantic-schema.py's own `Resident.address: Address` shape:
/// `Address` is a class, not a `type` alias, so `declared_refinement`'s
/// bare-Name arm reads nothing for it — `class_model_of`'s own
/// `.or_else` fallback must build `Address`'s member table instead.
/// `Resident`'s field carries `members: Some(...)` with `zip_code`'s
/// own declared set, so a later `judge_construction`/MEMBERS LAW
/// judgment of a nested dict can see past the bare class name.
#[test]
fn class_model_of_reads_a_field_annotated_with_another_module_level_class() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field, BaseModel\n",
        "type ZipCode = Annotated[str, Field(min_length=5, max_length=5)]\n",
        "class Address(BaseModel):\n",
        "    zip_code: ZipCode\n",
        "class Resident(BaseModel):\n",
        "    address: Address\n",
    ));
    let aliases = crate::surface::compile_aliases(&module);
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let resident = table.get("Resident").expect("Resident class recorded");
    let address_field = resident.fields.iter().find(|field| field.name == "address").expect("address field present");
    let declared = address_field.declared.as_ref().expect("Address reads as a member-carrying declaration");
    let members = declared.members.as_ref().expect("a class-typed field carries a per-member table");
    let zip_code = members.iter().find(|member| member.name == "zip_code").expect("zip_code member present");
    assert_eq!(zip_code.declared.spelling, "ZipCode");
}

/// The same shape one level deeper: `Resident.person: Person` where
/// `Person` is ITSELF a BaseModel with a refined field — nested
/// membership recurses because `Person` was built through the same
/// lazy `build_class_model` call, so its own `declared` already
/// carries `members: Some(...)` by the time `Resident`'s field reads it.
#[test]
fn class_model_of_reads_a_doubly_nested_member_class() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field, BaseModel\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Person(BaseModel):\n",
        "    age: Age\n",
        "class Resident(BaseModel):\n",
        "    person: Person\n",
    ));
    let aliases = crate::surface::compile_aliases(&module);
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let resident = table.get("Resident").expect("Resident class recorded");
    let person_field = resident.fields.iter().find(|field| field.name == "person").expect("person field present");
    let declared = person_field.declared.as_ref().expect("Person reads as a member-carrying declaration");
    let members = declared.members.as_ref().expect("a class-typed field carries a per-member table");
    let age = members.iter().find(|member| member.name == "age").expect("age member present");
    assert_eq!(age.declared.spelling, "Age");
}

/// A field annotated with a class name the module never declares
/// (a typo, or a class defined in another module this table cannot
/// see) declines exactly as before this unit — `declared: None`,
/// never a guessed member table.
#[test]
fn class_model_of_field_annotated_with_an_unknown_name_stays_undeclared() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from pydantic import BaseModel\n",
        "class Resident(BaseModel):\n",
        "    address: Missing\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let resident = table.get("Resident").expect("Resident class recorded");
    let address_field = resident.fields.iter().find(|field| field.name == "address").expect("address field present");
    assert!(address_field.declared.is_none(), "an undeclared class name states nothing this table reads");
}
