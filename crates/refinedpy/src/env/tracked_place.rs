//! Access-path bindings: a fact recorded about a PATH — a base binding
//! plus a chain of attribute segments (`a.n`, `d.tzinfo`) — rather than
//! a bare name. Mirrors the Go adapter's `dataflowfacts.TrackedPlace`
//! one-for-one (`mod.rs`'s own doc on why this exists).

use refined_domain::abstract_value::AbstractValue;

use super::Environment;

use super::TemporalOffsetDerivation;

impl Environment {
    /// Records that `name` holds a temporal offset of another name —
    /// `TemporalOffsetDerivation`'s own doc for what the three fields
    /// mean and why the ledger exists.
    pub fn record_temporal_offset(&mut self, name: &str, derivation: TemporalOffsetDerivation) {
        self.temporal_offsets.insert(name.to_owned(), derivation);
    }

    /// Every name currently recorded as a temporal offset of
    /// `instant_name`, paired with its derivation — what the narrowing
    /// channel re-derives once that instant's own window tightens.
    pub fn temporal_offsets_of(&self, instant_name: &str) -> Vec<(String, TemporalOffsetDerivation)> {
        self.temporal_offsets
            .iter()
            .filter(|(_, derivation)| derivation.instant_name == instant_name)
            .map(|(name, derivation)| (name.clone(), derivation.clone()))
            .collect()
    }

    /// Drops `name`'s own derivation — a write to the name replaces
    /// whatever it held, so the tie to the earlier instant no longer
    /// stands. Called from the same write channels that `forget` a
    /// binding.
    pub fn forget_temporal_offset(&mut self, name: &str) {
        self.temporal_offsets.remove(name);
    }
}

/// A tracked place: a base binding name plus a chain of segments — `a`
/// alone, `a.n`, `a.n.x` for a deeper attribute chain, or `v[0]` for a
/// LITERAL-INDEX read. Mirrors the Go adapter's
/// `dataflowfacts.TrackedPlace` (`Binding` + `Path []string`)
/// one-for-one, including that type's own bracket convention for an
/// element slot: an attribute segment is spelled `n`, an index segment
/// `[0]`, so `v[0]` is `TrackedPlace { binding: "v", path: ["[0]"] }`
/// and the two segment kinds never collide (a Python identifier can
/// never start with `[`).
///
/// The index segment is what makes a STABLE READ narrowable: `v[0]`
/// tested by a guard and `v[0]` read in that guard's branch name the
/// same place, so the guard's own narrowing is what the read answers
/// from — the same way `a.n` tested and `a.n` read already do. The
/// invariant that makes this sound is the one forget resolver: a write
/// to `v` drops every path fact rooted at `v` (`forget_path_base`), so
/// a fact only ever survives while the base is genuinely unwritten
/// between the guard and the read.
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

    /// `self` extended by one INDEX segment — `v.extend_index("0")`
    /// names `v[0]`. Spelled with the brackets so an index segment and
    /// an attribute segment of the same text stay distinct places
    /// (`d.0` is not `d[0]`), and so `words()` reads back as the source
    /// itself spells the read.
    pub fn extend_index(&self, index: &str) -> TrackedPlace {
        self.extend(&format!("[{index}]"))
    }

    /// This place spelled as one string — `a`, `a.n`, `a.n.x`, `v[0]`,
    /// `a.n[0]`. The one spelling the binding ledger keys by, so the
    /// write that files a derivation and the read that reclaims it name
    /// the same place by construction. An INDEX segment already carries
    /// its own brackets, so it joins with no separating dot and the
    /// spelling reads back exactly as the source writes the read.
    pub fn words(&self) -> String {
        let mut spelled = self.binding.clone();
        for segment in &self.path {
            match segment.starts_with('[') {
                true => spelled.push_str(segment),
                false => {
                    spelled.push('.');
                    spelled.push_str(segment);
                }
            }
        }
        spelled
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

/// `a.n.x` reads as `TrackedPlace { binding: "a", path: ["n", "x"] }`
/// and `v[0]` as `TrackedPlace { binding: "v", path: ["[0]"] }` — a bare
/// `Expr::Name` alone, a chain of `Expr::Attribute` reads over one, or a
/// LITERAL-index `Expr::Subscript` over one, all the way down to that
/// base name. Any other root (a call, a literal) names no place at all:
/// the checker cannot say the chain survives past a shape this reader
/// does not recognize.
///
/// Only a subscript whose index is written as a plain literal — a
/// non-negative integer (`v[0]`) or a string (`d["code"]`) — names a
/// place. A computed index (`v[i]`, `v[n + 1]`) does not: two reads
/// spelled the same way can select different elements, so the reads are
/// not the same place and a fact recorded at one would not be a fact
/// about the other. A negative index is likewise not read: `v[-1]`
/// selects by the sequence's own length, which is not fixed by the
/// spelling.
pub fn tracked_place_of(expression: &ruff_python_ast::Expr) -> Option<TrackedPlace> {
    match expression {
        ruff_python_ast::Expr::Name(name) => Some(TrackedPlace::bare(name.id.as_str())),
        ruff_python_ast::Expr::Attribute(attribute) => {
            let base = tracked_place_of(attribute.value.as_ref())?;
            Some(base.extend(attribute.attr.as_str()))
        }
        ruff_python_ast::Expr::Subscript(subscript) => {
            let base = tracked_place_of(subscript.value.as_ref())?;
            let index = literal_index_segment(subscript.slice.as_ref())?;
            Some(base.extend_index(&index))
        }
        _ => None,
    }
}

/// The index a subscript's own slice expression spells LITERALLY, as the
/// text an index segment carries — the non-negative integer `0` for
/// `v[0]`, the quoted key `"code"` for `d["code"]`. `None` for every
/// other slice (a name, an arithmetic expression, a slice range, a
/// negative literal): those select an element the spelling alone does
/// not fix, so two reads written identically are not the same place.
fn literal_index_segment(slice: &ruff_python_ast::Expr) -> Option<String> {
    match slice {
        ruff_python_ast::Expr::NumberLiteral(literal) => match &literal.value {
            ruff_python_ast::Number::Int(int) => match int.as_i64() {
                Some(value) if value >= 0 => Some(value.to_string()),
                _ => None,
            },
            _ => None,
        },
        ruff_python_ast::Expr::StringLiteral(literal) => Some(format!("{:?}", literal.value.to_str())),
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
