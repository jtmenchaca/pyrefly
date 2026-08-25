use super::*;
use crate::collection_models::list_literal_value;

#[test]
fn a_return_inside_a_for_loop_body_fires_at_the_carried_range() {
    let Some(kernel) = loaded_kernel() else { return };
    // c-reads-and-values.py:927/928's own shape: `for age in
    // overs.values(): return age` — every iterate is known, and the
    // loop's own answer must carry the returned value out so
    // walk_loop can judge it against -> Age, exactly as walk_return
    // would for a straight-line return.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    overs = {\"bea\": 200}\n",
        "    for age in overs.values():\n",
        "        return age\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "the loop must still run concretely — the return channel must not decline it: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the returned 200 must fire against the declared -> Age return: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_conditional_return_inside_a_loop_joins_the_return_path_with_the_normal_completion() {
    let Some(kernel) = loaded_kernel() else { return };
    // the return sits under an `if` that only SOME iterations take
    // (age == 200 never occurs here, so the loop actually completes
    // normally on every iteration and the return path never fires) —
    // this pins that the join keeps the NORMAL completion path alive
    // and does not wrongly treat "a return exists somewhere in the
    // body" as "every path returns."
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    total: Age = 0\n",
        "    for age in [10, 20]:\n",
        "        if age == 999:\n",
        "            return age\n",
        "        total = total + age\n",
        "    return total\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "every iterate stays in range on both the conditional-return and the normal-completion path: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn a_conditional_return_inside_a_loop_that_does_fire_judges_at_the_carried_range() {
    let Some(kernel) = loaded_kernel() else { return };
    // the SAME conditional shape, but the guarded return DOES trigger
    // on one iteration — the returned value must still fire.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    for age in [10, 200]:\n",
        "        if age > 100:\n",
        "            return age\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the conditional return's own out-of-set value (200) must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

// --- BODY-LOCAL CLASS TABLES ---

/// A class defined INSIDE a function body (b-body-expressions.py's
/// `new_resolvable` shape): `class_table`'s own module-level scan
/// never sees it, so before this fix a body-local construction
/// stayed `unknown()` and the fire never landed. `merged_classes_for_body`
/// now merges this body's own top-level classes over `context.classes`,
/// so `Person(200)`'s field carries the summary into the return sink.
#[test]
fn a_class_defined_inside_a_function_body_still_judges_its_construction() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    class Person:\n",
        "        def __init__(self, age: int) -> None:\n",
        "            self.age = age\n",
        "    ok = Person(40)\n",
        "    good: Age = ok.age\n",
        "    over = Person(200)\n",
        "    return over.age\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the body-local class's own out-of-set construction (200) must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// c-reads-and-values.py's own `read_one_field`/`read_nested_path`
/// collision: two DIFFERENT functions each declare their own body-local
/// class named `Person`, with different fields. Both classes collide
/// under the one shared bare name in `context.classes`
/// (`findings_for_module_with_resolver`'s own module-wide scan), so
/// `construction_call_verdict` must read `environment.classes()` (the
/// per-body table `merged_classes_for_body` built for THIS body) rather
/// than `context.classes` alone — otherwise `Person(age=40)` matches
/// whichever class happened to overwrite the shared entry, not the
/// caller's own local `Person`.
#[test]
fn a_body_local_class_construction_uses_its_own_bodys_class_not_a_same_named_sibling() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import BaseModel, Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def read_one_field() -> Age:\n",
        "    class Person(BaseModel):\n",
        "        age: int\n",
        "    over = Person(age=200)\n",
        "    return over.age\n",
        "def other_function_with_same_named_class() -> None:\n",
        "    class Person(BaseModel):\n",
        "        name: str\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "the same-named sibling class must not shadow this body's own Person: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "read_one_field's own Person(age=200) must fire through its own body-local class: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

// --- SELF-SEEDING ---

/// `self.age` read inside a method body, with NO call site anywhere
/// in the module (b-body-expressions.py's `self_field_read`/
/// `OverPerson` shape) — before this fix, `self` was never bound
/// during the STATEMENT WALK of a method body (only `method_
/// call_result`'s separate call-site interpreter seeded it), so this
/// read answered `Unknown` and stayed silent. `walk_method_def` now
/// seeds `self` from the class's own declared/default field shape at
/// the method body's own entry, so the literal self-write inside
/// `__init__` (captured as the field's DEFAULT, `class_table`'s own
/// literal-self-write rule) carries into `years`'s own `self.age`
/// read and judges against the method's `-> Age` annotation.
#[test]
fn a_self_field_read_inside_a_method_body_judges_with_no_call_site() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class OverPerson:\n",
        "    def __init__(self) -> None:\n",
        "        self.age = 200\n",
        "    def years(self) -> Age:\n",
        "        return self.age\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "self.age's own out-of-set default (200) must fire at the method's own return: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// A bare `self` reference judges too (b-body-expressions.py's
/// `ThisBare` shape): an Object value against a scalar-ground
/// declared set is `assignability.rs`'s own "Object/List/Null vs
/// scalar-ground → Fire" law — reachable only once `self` is bound
/// to something at all.
#[test]
fn a_bare_self_reference_fires_against_a_scalar_ground_return_annotation() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Bare:\n",
        "    def years(self) -> Age:\n",
        "        return self\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "a bare self reference is not a refined Age: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

// --- setdefault_append (dict_groupby's chained mutation) ---

#[test]
fn setdefault_append_extends_a_present_key_and_writes_a_new_one() {
    use refined_domain::abstract_value::{known_values, PrimitiveKind};
    use refined_domain::trust_grades::TrustProved;
    fn integer(v: f64) -> AbstractValue {
        known_values(vec![v], PrimitiveKind::Integer, TrustProved)
    }
    fn string(text: &str) -> AbstractValue {
        let code_points: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
        known_values(code_points, PrimitiveKind::String, TrustProved)
    }
    let grouped = crate::collection_models::dict_literal_value(
        &[Some(crate::collection_models::DictKey::string("young"))],
        &[list_literal_value(&[integer(40.0)])],
    );
    // "young" is present: setdefault_append reads its existing list
    // and appends onto it, rather than replacing with the default.
    let after_young = setdefault_append(&grouped, &string("young"), &list_literal_value(&[]), &integer(41.0))
        .expect("appending onto a present key's list must decide");
    assert_eq!(
        crate::collection_models::subscript_read(&after_young, &string("young")),
        Some(list_literal_value(&[integer(40.0), integer(41.0)]))
    );
    // "old" is absent: setdefault_append inserts the default list,
    // then appends onto that fresh list — the exact
    // `grouped.setdefault("old", []).append(200)` shape.
    let after_old = setdefault_append(&after_young, &string("old"), &list_literal_value(&[]), &integer(200.0))
        .expect("appending onto a fresh default list must decide");
    assert_eq!(
        crate::collection_models::subscript_read(&after_old, &string("old")),
        Some(list_literal_value(&[integer(200.0)]))
    );
}
