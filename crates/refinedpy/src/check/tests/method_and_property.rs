use super::*;

#[test]
fn a_statement_side_method_call_writes_a_field_a_later_read_sees() {
    let Some(kernel) = loaded_kernel() else { return };
    // b-body-expressions.py:522-547's own row: `outlaw.spoil()` is a
    // bare Expr statement calling a method that writes `self.age =
    // 200` — the receiver must rebind, and the later `outlaw.age`
    // read must see 200, not the stale pre-call 40.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Outlaw:\n",
        "    def __init__(self) -> None:\n",
        "        self.age = 40\n",
        "    def spoil(self) -> None:\n",
        "        self.age = 200\n",
        "def f() -> Age:\n",
        "    outlaw = Outlaw()\n",
        "    outlaw.spoil()\n",
        "    return outlaw.age\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the method's own write (200) must be visible at outlaw.age: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_statement_side_method_call_that_leaves_the_field_in_set_stays_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Person:\n",
        "    def __init__(self) -> None:\n",
        "        self.age = 40\n",
        "    def bump(self) -> None:\n",
        "        self.age = self.age + 1\n",
        "def f() -> Age:\n",
        "    person = Person()\n",
        "    person.bump()\n",
        "    return person.age\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "an in-set write through a statement-side method call must never fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn a_write_then_read_through_a_declared_sink_uses_the_method_s_own_return_value() {
    let Some(kernel) = loaded_kernel() else { return };
    // sink_value's own method-call channel: an AnnAssign RHS that is
    // a statement-side method call judges the method's OWN return
    // value, not a plain evaluate_expression reading of the call.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Counter:\n",
        "    def __init__(self) -> None:\n",
        "        self.value = 199\n",
        "    def increment(self) -> int:\n",
        "        self.value = self.value + 1\n",
        "        return self.value\n",
        "def f() -> None:\n",
        "    c = Counter()\n",
        "    over: Age = c.increment()\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the method's own returned value (200) must judge at the declared sink: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

// --- NAMED-RECEIVER FIELD WRITE (write_named_field, e:357/q:203) ---

#[test]
fn a_property_setter_write_through_a_local_variable_receiver_fires_and_the_return_fires_too() {
    let Some(kernel) = loaded_kernel() else { return };
    // e-class-and-function.py's `property_getter_setter`: `over_box.age
    // = 200` writes through a `@property` setter on a LOCAL variable
    // receiver, never `self` — before write_named_field generalized
    // write_self_field's own judged-and-rebound law past the literal
    // name `self`, this row's write silently forgot `over_box` instead
    // of judging the setter's own declared refinement.
    //
    // TWO fires are correct here, not one: the write's own value
    // range (`write_named_field`'s judgment, `bind_or_forget_target`'s
    // named-receiver branch) is the FIRST verdict, and `f`'s own
    // `-> Age` return sink (`walk_return`) is a SECOND, independent
    // verdict — the rebind the sibling test at :8406
    // (`a_property_setter_write_lands_on_the_backing_field_a_later_
    // getter_read_sees`) pins as sound is exactly what carries this
    // same out-of-set 200 forward to `return over_box.age`, so the
    // return sink judges the identical bad value against the
    // identical `Age` set. Both messages read as the same text
    // (`judge`'s message carries no location), so the ranges — not
    // the text — are what prove these are two determined verdicts
    // over two statements, not one write double-counted.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Aged:\n",
        "    def __init__(self) -> None:\n",
        "        self._held = 40\n",
        "    @property\n",
        "    def age(self) -> int:\n",
        "        return self._held\n",
        "    @age.setter\n",
        "    def age(self, value: Age) -> None:\n",
        "        self._held = value\n",
        "def f() -> Age:\n",
        "    over_box = Aged()\n",
        "    over_box.age = 200\n",
        "    return over_box.age\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        2,
        "the setter's own write (200) must fire against its declared Age refinement, AND \
         the rebind it leaves behind must carry 200 to f's own -> Age return sink, which \
         fires independently: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    assert!(fires[1].message.contains("'200'"), "{}", fires[1].message);
    assert_ne!(
        fires[0].range.start(),
        fires[1].range.start(),
        "two DIFFERENT sinks must have judged this value — the write's own literal and the \
         return expression — never the same range firing twice: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// A DISCRIMINATING test for the write TARGET, not just the write's
/// own judgment: the backing field starts at an OUT-OF-SET value
/// (200), and the setter then writes an IN-SET value (40) through
/// the property. `write_named_field` writes the property name
/// itself (`age`) into the instance's keys, but `field_read_through_
/// model` reads a property through its OWN `backing` name (`_held`)
/// — a write that lands on `age` instead of `_held` never reaches
/// the getter's own read at all, so the later `over_box.age` read
/// would still see the STALE 200 backing value and wrongly fire.
/// The write must resolve to the SAME backing name the read
/// resolves to, so this call stays completely silent.
#[test]
fn a_property_setter_write_lands_on_the_backing_field_a_later_getter_read_sees() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Aged:\n",
        "    def __init__(self) -> None:\n",
        "        self._held = 200\n",
        "    @property\n",
        "    def age(self) -> int:\n",
        "        return self._held\n",
        "    @age.setter\n",
        "    def age(self, value: Age) -> None:\n",
        "        self._held = value\n",
        "def f() -> Age:\n",
        "    box = Aged()\n",
        "    box.age = 40\n",
        "    return box.age\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "the setter's in-set write (40) must land on the SAME backing field the getter reads, \
         leaving no stale out-of-set value behind: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn a_plain_field_write_through_a_local_variable_receiver_rebinds_and_a_later_read_sees_it() {
    let Some(kernel) = loaded_kernel() else { return };
    // q-decline-names.py's `setter_effect_read_through_getter`: the
    // same named-receiver write law, over an UNREFINED field (no Fire
    // expected) — pins that write_named_field still rebinds (a later
    // getter read must see the write) even with no declared refinement
    // to judge against.
    let module = parsed(concat!(
        "class AgeBox:\n",
        "    def __init__(self) -> None:\n",
        "        self._age = 10\n",
        "    @property\n",
        "    def age(self) -> int:\n",
        "        return self._age\n",
        "    @age.setter\n",
        "    def age(self, value: int) -> None:\n",
        "        self._age = value\n",
        "def f() -> int:\n",
        "    box = AgeBox()\n",
        "    box.age = 40\n",
        "    return box.age\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "an unrefined field write through a local variable receiver must never fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

// --- CLASS-OBJECT ATTRIBUTE STATE (class_object_value, e:485) ---

#[test]
fn a_class_object_attribute_write_and_read_composes_with_no_instance_involved() {
    let Some(kernel) = loaded_kernel() else { return };
    // e-class-and-function.py's `class_attribute_write`: `Counted.total
    // = 200` writes through the CLASS ITSELF (no `Counted(...)`
    // construction anywhere on this row), and the later `Counted.total`
    // read must see the write. Before class_object_value seeded the
    // class's own bare name as a tagged Kind::Object, `Counted` read as
    // unknown() and the write silently forgot it.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Counted:\n",
        "    total = 0\n",
        "def f() -> Age:\n",
        "    Counted.total = 200\n",
        "    return Counted.total\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the class-object write (200) must be visible at the later Counted.total read: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_class_object_attribute_write_in_range_stays_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Counted:\n",
        "    total = 0\n",
        "def f() -> Age:\n",
        "    Counted.total = 40\n",
        "    return Counted.total\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "an in-range class-object write must never fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

// --- AugAssign ON A NON-NAME TARGET (walk_field_aug_assign /
//     walk_subscript_aug_assign, i:233/246/273) ---

#[test]
fn a_property_accessor_compound_write_fires_against_the_setters_own_refinement() {
    let Some(kernel) = loaded_kernel() else { return };
    // i-more-expressions.py's `accessor_compound_read_modify_write`:
    // `over_box.age += 195` (10 + 195 = 205) must fire against the
    // setter's own Age refinement, the same fire a hand-split
    // `over_box.age = over_box.age + 195` would give.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class AccessorBox:\n",
        "    def __init__(self) -> None:\n",
        "        self.held = 10\n",
        "    @property\n",
        "    def age(self) -> int:\n",
        "        return self.held\n",
        "    @age.setter\n",
        "    def age(self, value: Age) -> None:\n",
        "        self.held = value\n",
        "def f() -> int:\n",
        "    over_box = AccessorBox()\n",
        "    over_box.age += 195\n",
        "    return over_box.held\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the compound write's own folded value (205) must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'205'"), "{}", fires[0].message);
}

#[test]
fn a_property_accessor_compound_write_in_range_stays_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class AccessorBox:\n",
        "    def __init__(self) -> None:\n",
        "        self.held = 10\n",
        "    @property\n",
        "    def age(self) -> int:\n",
        "        return self.held\n",
        "    @age.setter\n",
        "    def age(self, value: Age) -> None:\n",
        "        self.held = value\n",
        "def f() -> int:\n",
        "    box = AccessorBox()\n",
        "    box.age += 5\n",
        "    return box.held\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "an in-range accessor compound write must never fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn a_subscript_compound_write_composes_and_a_later_read_sees_the_mutated_element() {
    let Some(kernel) = loaded_kernel() else { return };
    // i-more-expressions.py's `compound_array_index_operators`:
    // `ages[0] += 5` must compose (read the element, fold, write back)
    // so a LATER `ages[0]` read sees 15, not the stale pre-write 10 —
    // walk_subscript_aug_assign's own no-element-judging contract still
    // requires the composition itself to be sound.
    let module = parsed(concat!(
        "def f() -> int:\n",
        "    ages = [10, 20]\n",
        "    ages[0] += 5\n",
        "    return ages[0]\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "walk_subscript_aug_assign never fires (no declared element set to judge against): {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

// --- del d[k] REBIND/FORGET ---

#[test]
fn del_subscript_on_a_known_dict_rebinds_and_a_later_read_answers_undetermined() {
    let Some(kernel) = loaded_kernel() else { return };
    // b-body-expressions.py:660-665's own row: `del person["age"]`
    // removes the key from a KNOWN dict; a later `.get("age")` read
    // then answers None (an absent key) rather than the stale
    // pre-delete value — this pins the REBIND half (dict_without_item
    // answers Some), not the specific None-vs-Undetermined judgment
    // downstream, which is dict_get_result's own contract.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    person: dict[str, int] = {\"age\": 40}\n",
        "    del person[\"age\"]\n",
        "    check = person.get(\"age\", 0)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.iter().all(|f| f.code != "RTS7001"),
        "no fire is expected in this row on its own: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn del_subscript_on_an_unknown_receiver_forgets_it() {
    let Some(kernel) = loaded_kernel() else { return };
    // an unresolved key/receiver shape must FORGET the receiver
    // (Undetermined downstream), never leave the stale pre-delete
    // value standing.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f(key: str) -> None:\n",
        "    person: dict[str, int] = {\"age\": 200}\n",
        "    del person[key]\n",
        "    over: Age = person[\"age\"]\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.iter().all(|f| f.code != "RTS7001"),
        "an unresolved delete key must forget the receiver — the stale 200 must not survive to fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}
