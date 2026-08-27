use super::*;

fn provable_raise_of(source: &str) -> Option<(TextRange, String)> {
    let kernel = loaded_kernel()?;
    let parsed = parse_expression(source).expect("test source must parse");
    let environment = empty_environment();
    provable_raise(&parsed.into_expr(), &environment, &kernel)
}

#[test]
fn test_provable_raise_zero_division() {
    let Some(found) = provable_raise_of("1 / 0") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("1 / 0 must provably raise");
    };
    assert!(found.1.contains("ZeroDivisionError"), "{}", found.1);
    assert!(found.1.contains("division by zero"), "{}", found.1);
}

#[test]
fn test_provable_raise_zero_floor_division_and_modulo() {
    let Some(found) = provable_raise_of("1 // 0") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("1 // 0 must provably raise");
    };
    assert!(found.1.contains("ZeroDivisionError"), "{}", found.1);

    let Some(found) = provable_raise_of("1 % 0") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("1 % 0 must provably raise");
    };
    assert!(found.1.contains("ZeroDivisionError"), "{}", found.1);
}

#[test]
fn test_provable_raise_out_of_range_subscript() {
    let Some(found) = provable_raise_of("[1, 2][5]") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("[1, 2][5] must provably raise");
    };
    assert!(found.1.contains("IndexError"), "{}", found.1);
}

#[test]
fn test_provable_raise_absent_dict_key() {
    let Some(found) = provable_raise_of("{\"a\": 1}[\"missing\"]") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("a missing dict key must provably raise");
    };
    assert!(found.1.contains("KeyError"), "{}", found.1);
}

#[test]
fn test_provable_raise_absent_int_dict_key() {
    // the key reader this row shares with the construction side covers
    // every key sort that can build an entry, not exact strings alone
    let Some(found) = provable_raise_of("{15: 1}[16]") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("a missing int dict key must provably raise");
    };
    assert!(found.1.contains("KeyError"), "{}", found.1);
    assert!(found.1.contains("16"), "{}", found.1);
}

#[test]
fn test_provable_raise_int_of_unparseable_string() {
    let Some(found) = provable_raise_of("int(\"xyz\")") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("int(\"xyz\") must provably raise");
    };
    assert!(found.1.contains("ValueError"), "{}", found.1);
    assert!(found.1.contains("invalid literal"), "{}", found.1);
}

#[test]
fn test_provable_raise_int_of_valid_string_declines() {
    assert!(provable_raise_of("int(\"123\")").is_none());
    // the underscore-digit-separator row (functions.rst) must NOT
    // false-positive raise
    assert!(provable_raise_of("int(\"1_000\")").is_none());
}

#[test]
fn test_provable_raise_string_index_miss() {
    let Some(found) = provable_raise_of("\"banana\".index(\"z\")") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("a missing needle's .index() must provably raise");
    };
    assert!(found.1.contains("ValueError"), "{}", found.1);
}

#[test]
fn test_provable_raise_list_index_miss() {
    let Some(found) = provable_raise_of("[1, 2, 3].index(9)") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("a missing element's .index() must provably raise");
    };
    assert!(found.1.contains("ValueError"), "{}", found.1);
}

#[test]
fn test_provable_raise_bytes_out_of_range_index() {
    let Some(found) = provable_raise_of("b\"ab\"[10]") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("an out-of-range bytes index must provably raise");
    };
    assert!(found.1.contains("IndexError"), "{}", found.1);
    // the message speaks in provable_raise's own voice, not
    // bytes_models.rs's "this read provably raises" wording
    assert!(found.1.starts_with("this expression provably raises"), "{}", found.1);
}

#[test]
fn test_provable_raise_string_out_of_range_subscript() {
    let Some(found) = provable_raise_of("\"banana\"[99]") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("an out-of-range string subscript must provably raise");
    };
    assert!(found.1.contains("IndexError"), "{}", found.1);
    assert!(found.1.contains("string index out of range"), "{}", found.1);
}

#[test]
fn test_provable_raise_string_in_range_subscript_declines() {
    assert!(provable_raise_of("\"banana\"[0]").is_none());
    // a negative in-range index must not false-positive raise
    assert!(provable_raise_of("\"banana\"[-1]").is_none());
}

#[test]
fn test_provable_raise_none_case() {
    assert!(provable_raise_of("1 + 2").is_none());
    assert!(provable_raise_of("[1, 2][0]").is_none());
    assert!(provable_raise_of("1 / 2").is_none());
}

#[test]
fn test_provable_raise_math_sqrt_of_known_negative() {
    let Some(found) = provable_raise_of("math.sqrt(-2)") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("math.sqrt(-2) must provably raise");
    };
    assert!(found.1.contains("ValueError"), "{}", found.1);
    assert!(found.1.contains("math domain error"), "{}", found.1);
}

#[test]
fn test_provable_raise_math_sqrt_of_known_nonnegative_declines() {
    assert!(provable_raise_of("math.sqrt(4)").is_none());
}

/// `date.fromisoformat("2023-02-29")`/`"2023-04-31"` — syntactically
/// `YYYY-MM-DD`-shaped but calendrically invalid (2023 is not a leap
/// year; April has 30 days). `test_date_fromisoformat_of_a_
/// calendrically_invalid_string_declines` already pins the VALUE side
/// (`Kind::Unknown`); this pins the matching RAISE-side determination
/// `date_fromisoformat_raises` adds.
#[test]
fn test_provable_raise_date_fromisoformat_of_a_calendrically_invalid_string() {
    for source in ["datetime.date.fromisoformat(\"2023-02-29\")", "datetime.date.fromisoformat(\"2023-04-31\")"] {
        let Some(found) = provable_raise_of(source) else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("{source} must provably raise");
        };
        assert!(found.1.contains("ValueError"), "{source}: {}", found.1);
    }
}

/// `date.fromisoformat("13:45")` — a clock time, not a date: no
/// hyphens at all, so the strict `YYYY-MM-DD` split never even
/// produces three parts. CPython's own `fromisoformat` raises
/// `ValueError` on any string outside its accepted grammar, the same
/// as a calendrically invalid one — `date_fromisoformat_raises`'s own
/// doc states why both shapes fire the identical row.
#[test]
fn test_provable_raise_date_fromisoformat_of_a_non_date_string() {
    let Some(found) = provable_raise_of("datetime.date.fromisoformat(\"13:45\")") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("datetime.date.fromisoformat(\"13:45\") must provably raise");
    };
    assert!(found.1.contains("ValueError"), "{}", found.1);
}

#[test]
fn test_provable_raise_date_fromisoformat_of_a_valid_string_declines() {
    assert!(provable_raise_of("datetime.date.fromisoformat(\"2024-03-01\")").is_none());
}

/// `math.log(-2)`/`math.log2(-2)`/`math.log10(-2)`: a KNOWN operand
/// entirely inside CPython's raise domain (`x <= 0`) fires the
/// determined "math domain error" finding, one shared row per
/// `DomainLimitedFamily::of_function`.
#[test]
fn test_provable_raise_math_log_family_of_a_known_nonpositive() {
    for source in ["math.log(-2)", "math.log2(-2)", "math.log10(-2)"] {
        let Some(found) = provable_raise_of(source) else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("{source} must provably raise");
        };
        assert!(found.1.contains("ValueError"), "{source}: {}", found.1);
        assert!(found.1.contains("math domain error"), "{source}: {}", found.1);
    }
}

/// `math.log(0.0)` provably raises — the module's own worked
/// example (specifications/python/Doc/library/math.rst:696-698) and
/// the exact JS/Python divergence point: the kernel's own `js.log`
/// arm serves `-inf` there (JavaScript's `Math.log(0) ===
/// -Infinity`), but CPython's `loghelper`/`math_1` (mathmodule.c)
/// raises `ValueError` for an infinite result from a finite input.
#[test]
fn test_provable_raise_math_log_of_exact_zero_the_python_javascript_divergence_point() {
    let Some(found) = provable_raise_of("math.log(0.0)") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("math.log(0.0) must provably raise — the module's own worked ValueError example");
    };
    assert!(found.1.contains("ValueError"), "{}", found.1);
    assert!(found.1.contains("math domain error"), "{}", found.1);
}

/// `math.log1p(-2.0)` (entirely `x <= -1`) provably raises; the
/// exact boundary point `math.log1p(-1.0)` ALSO raises (the closed
/// `x <= -1` domain, not the kernel's open `x < -1` NaN corner) —
/// `jsLog1p` serves `-inf` there, another JS/Python divergence.
#[test]
fn test_provable_raise_math_log1p_of_nonpositive_and_its_exact_boundary() {
    for source in ["math.log1p(-2.0)", "math.log1p(-1.0)"] {
        let Some(found) = provable_raise_of(source) else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("{source} must provably raise");
        };
        assert!(found.1.contains("ValueError"), "{source}: {}", found.1);
        assert!(found.1.contains("math domain error"), "{source}: {}", found.1);
    }
}

/// `math.asin(2.0)`/`math.acos(-2.0)`: entirely outside `[-1, 1]`
/// fires the determined finding; `math.asin(1.0)` (the CLOSED
/// boundary) does NOT raise — `asin`/`acos`'s raise domain is the
/// OPEN ray `|x| > 1`, matching the kernel's own boundary exactly
/// (no JS/Python divergence for this family).
#[test]
fn test_provable_raise_math_asin_acos_outside_domain_and_boundary_declines() {
    let Some(found) = provable_raise_of("math.asin(2.0)") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("math.asin(2.0) must provably raise");
    };
    assert!(found.1.contains("math domain error"), "{}", found.1);

    let Some(found) = provable_raise_of("math.acos(-2.0)") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("math.acos(-2.0) must provably raise");
    };
    assert!(found.1.contains("math domain error"), "{}", found.1);

    assert!(provable_raise_of("math.asin(1.0)").is_none(), "asin(1.0) = pi/2 exactly — must not raise");
    assert!(provable_raise_of("math.acos(-1.0)").is_none(), "acos(-1.0) = pi exactly — must not raise");
}

/// `math.atanh(2.0)` (entirely `|x| >= 1`) provably raises; the
/// exact boundary points `math.atanh(1.0)`/`math.atanh(-1.0)` ALSO
/// raise (the closed `|x| >= 1` domain) — `jsAtanh` serves `±inf`
/// there, another JS/Python divergence this family's own raise
/// domain must be one ray WIDER than the kernel's boundary to catch.
#[test]
fn test_provable_raise_math_atanh_outside_domain_and_its_exact_boundary() {
    for source in ["math.atanh(2.0)", "math.atanh(1.0)", "math.atanh(-1.0)"] {
        let Some(found) = provable_raise_of(source) else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("{source} must provably raise");
        };
        assert!(found.1.contains("ValueError"), "{source}: {}", found.1);
        assert!(found.1.contains("math domain error"), "{source}: {}", found.1);
    }
}

/// `math.acosh(0.5)`: entirely `x < 1` fires; `math.acosh(1.0)` (the
/// CLOSED boundary, `acosh(1) = 0` exactly) does NOT raise —
/// `acosh`'s raise domain is the OPEN ray `x < 1`, matching the
/// kernel's own boundary exactly (no JS/Python divergence).
#[test]
fn test_provable_raise_math_acosh_below_one_and_boundary_declines() {
    let Some(found) = provable_raise_of("math.acosh(0.5)") else {
        if loaded_kernel().is_none() {
            return;
        }
        panic!("math.acosh(0.5) must provably raise");
    };
    assert!(found.1.contains("math domain error"), "{}", found.1);
    assert!(provable_raise_of("math.acosh(1.0)").is_none(), "acosh(1.0) = 0 exactly — must not raise");
}
