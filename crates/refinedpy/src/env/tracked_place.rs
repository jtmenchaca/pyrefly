//! Access-path bindings: a fact recorded about a PATH — a base binding
//! plus a chain of attribute segments (`a.n`, `d.tzinfo`) — rather than
//! a bare name. Mirrors the Go adapter's `dataflowfacts.TrackedPlace`
//! one-for-one (`mod.rs`'s own doc on why this exists).

use refined_domain::abstract_value::AbstractValue;

use super::Environment;

/// A tracked place: a base binding name plus a chain of attribute
/// segments — `a` alone, or `a.n`, or `a.n.x` for a deeper chain. Mirrors
/// the Go adapter's `dataflowfacts.TrackedPlace` (`Binding` + `Path
/// []string`) one-for-one, scoped down to the attribute-segment half
/// this file needs today: Python's own subscript syntax (`d["k"]`) is a
/// DIFFERENT construct (dict-presence narrowing, not an access-path
/// fact), so this carries no index-segment spelling the Go type's own
/// bracket convention gives its element slots.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TrackedPlace {
    pub binding: String,
    pub path: Vec<String>,
}

impl TrackedPlace {
    /// A bare binding, no path segments — `TrackedPlace::bare("a").path`
    /// is empty, matching the Go type's own `Path: nil` for a plain name.
    pub fn bare(binding: &str) -> TrackedPlace {
        TrackedPlace { binding: binding.to_owned(), path: Vec::new() }
    }

    /// `self` extended by one more attribute segment — `a.extend("n")`
    /// names `a.n`; `a.extend("n").extend("x")` names `a.n.x`.
    pub fn extend(&self, segment: &str) -> TrackedPlace {
        let mut path = self.path.clone();
        path.push(segment.to_owned());
        TrackedPlace { binding: self.binding.clone(), path }
    }

    /// Whether `self` is `prefix` itself, or a path that CONTINUES
    /// `prefix` with one or more further segments — the containment test
    /// `forget_path_base` uses: a write to `a.n` must also drop `a.n.x`
    /// (continues) and `a.n` itself (equal), but never an unrelated
    /// sibling path like `a.m`.
    pub fn extends(&self, prefix: &TrackedPlace) -> bool {
        self.binding == prefix.binding && self.path.len() >= prefix.path.len() && self.path[..prefix.path.len()] == prefix.path[..]
    }
}

/// `a.n.x` reads as `TrackedPlace { binding: "a", path: ["n", "x"] }` —
/// a bare `Expr::Name` alone, or a chain of `Expr::Attribute` reads over
/// one, all the way down to that base name. Any other root (a call, a
/// subscript, a literal) names no place at all: the checker cannot say
/// the chain survives past a shape this reader does not recognize.
pub fn tracked_place_of(expression: &ruff_python_ast::Expr) -> Option<TrackedPlace> {
    match expression {
        ruff_python_ast::Expr::Name(name) => Some(TrackedPlace::bare(name.id.as_str())),
        ruff_python_ast::Expr::Attribute(attribute) => {
            let base = tracked_place_of(attribute.value.as_ref())?;
            Some(base.extend(attribute.attr.as_str()))
        }
        _ => None,
    }
}

impl Environment {
    /// Record what an ACCESS PATH holds after a condition the walk
    /// narrowed it by (`narrowing.rs`'s own SET/VALUES channels, once a
    /// comparison's tested side is a path rather than a bare name).
    pub fn bind_path(&mut self, place: &TrackedPlace, value: AbstractValue) {
        self.path_bindings.insert(place.clone(), value);
    }

    /// What the access path holds here, if the walk bound it.
    pub fn read_path(&self, place: &TrackedPlace) -> Option<&AbstractValue> {
        self.path_bindings.get(place)
    }

    /// Drop every access-path fact this environment holds about `prefix`
    /// itself, or about any path CONTINUING `prefix` (`TrackedPlace::
    /// extends`'s own doc) — the one forget resolver: a write to `a`
    /// (`prefix` is the bare place `a`) drops `a.n` and `a.n.x` alike; a
    /// write to `a.n` (`prefix` is `a.n`) drops `a.n.x` too but leaves an
    /// unrelated sibling path (`a.m`) standing. Every write channel that
    /// can invalidate a path fact routes through this one function,
    /// rather than each reimplementing the prefix test.
    pub fn forget_path_base(&mut self, prefix: &TrackedPlace) {
        self.path_bindings.retain(|place, _| !place.extends(prefix));
    }
}
