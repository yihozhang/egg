//! Stochastic search backend for egg.
//!
//! This module provides an alternative to e-graph based equality saturation that
//! operates directly over [`RecExpr`]s without building an [`EGraph`].
//!
//! # Overview
//!
//! The central type is [`State`], which wraps a [`RecExpr`] together with
//! per-node analysis data (computed by a [`StoAnalysis`]) and subtree sizes.
//! Rewrite rules are expressed as [`StoRewrite`]s, whose left-hand side is a
//! [`StoSearcher`] and right-hand side is a [`StoApplier`].
//!
//! Unlike the e-graph approach, applying a rewrite appends new nodes to the
//! state rather than merging equivalence classes.  The state is periodically
//! compacted (via [`State::compact_keeping`]) to discard nodes no longer
//! reachable from any live root.
//!
//! # Metropolis-Hastings
//!
//! [`StoRunner`] implements Metropolis-Hastings over the space of terms.
//! At each step it randomly picks a rule, randomly picks a match, applies the
//! rewrite (via [`State::transplant`]), and accepts or rejects the result
//! according to the standard MH criterion using [`StoAnalysis::cost`] and a
//! [`BetaSchedule`].
//!
//! # Substitutions
//!
//! Pattern variables are bound to [`Id`]s that serve as *positions* in
//! `rec_expr` — the same role [`Id`]s play as e-class identifiers in the
//! e-graph interface.  The existing [`Subst`] type is reused directly.

use std::fmt::Debug;
use std::mem;
use std::sync::Arc;
use std::{collections::HashMap, fmt::Display};

use crate::{Id, Language, RecExpr, Subst, Symbol, Var};

// ─── StoAnalysis ─────────────────────────────────────────────────────────────

/// Analysis for stochastic rewriting over a [`RecExpr`].
///
/// Unlike [`Analysis`][crate::Analysis], which is tightly coupled to the
/// [`EGraph`][crate::EGraph], `StoAnalysis` computes data purely bottom-up:
/// the data for a node is derived from the node itself and the already-computed
/// data of its children.  This makes it suitable for direct use on a flat
/// [`RecExpr`] without an e-graph.
///
/// # Example
///
/// ```rust,ignore
/// struct ConstFold;
///
/// impl StoAnalysis<Math> for ConstFold {
///     type Data = Option<i64>;
///
///     fn make(enode: &Math, analysis: &[Option<i64>]) -> Option<i64> {
///         match enode {
///             Math::Num(n) => Some(*n),
///             Math::Add([a, b]) => Some(analysis[usize::from(*a)]? + analysis[usize::from(*b)]),
///             _ => None,
///         }
///     }
/// }
/// ```
pub trait StoAnalysis<L: Language>: Sized {
    /// The per-node data produced by this analysis.
    type Data: Debug + Clone;

    /// Compute analysis data for the node at the current insertion point.
    ///
    /// `analysis` holds the data for all nodes already in the state
    /// (indices `0..current`).  Every child [`Id`] of `enode` is a valid
    /// index into `analysis`.
    fn make(enode: &L, analysis: &[Self::Data]) -> Self::Data;

    /// Compute the cost of the node at the current insertion point.
    ///
    /// Both `analysis` and `children_cost` hold entries for all nodes already
    /// in the state (indices `0..current`), so every child [`Id`] of `enode`
    /// is a valid index into either slice.
    ///
    /// The default implementation returns AST node count (1 + sum of children
    /// costs), matching [`egg::AstSize`][crate::AstSize].
    fn cost(enode: &L, _analysis: &[Self::Data], children_cost: &[f64]) -> f64 {
        1.0 + enode.fold(0.0, |acc, child| acc + children_cost[usize::from(child)])
    }

    /// Hook called after a node is appended to a [`State`].
    ///
    /// Can be used to trigger derived insertions (e.g., constant folding).
    /// The default does nothing.
    #[allow(unused_variables)]
    fn modify(state: &mut State<L, Self>, pos: Id) {}

    /// Remap any [`Id`]s stored inside `data` after a compaction.
    ///
    /// `remap[old_idx]` gives the new index for a node that was at `old_idx`
    /// before compaction.  Called for every surviving node's data entry.
    /// The default is a no-op (suitable when `Data` contains no `Id`s).
    #[allow(unused_variables)]
    fn remap_data(data: &mut Self::Data, remap: &[u32]) {}
}

/// No-op analysis; stores `()` for every node.
impl<L: Language> StoAnalysis<L> for () {
    type Data = ();
    fn make(_enode: &L, _analysis: &[()]) {}
}

// ─── State ───────────────────────────────────────────────────────────────────

/// The mutable state for stochastic term rewriting.
///
/// A `State` wraps a [`RecExpr`] (a flat, topologically-ordered list of nodes)
/// together with per-node analysis data and subtree sizes.  New nodes are
/// appended via [`State::add`]; dead nodes are removed by
/// [`State::compact_keeping`].
///
/// # Invariants
///
/// - `rec_expr.len() == analysis.len() == size.len() == cost.len()`
/// - For every node at index `i`, all child [`Id`]s satisfy `child < i`.
/// - `size[i]` equals the number of nodes in the subtree rooted at `i`.
/// - `cost[i]` equals [`A::cost`][StoAnalysis::cost] evaluated at `i`.
pub struct State<L: Language, A: StoAnalysis<L>> {
    /// The expression as a flat list; the root is the last element.
    pub rec_expr: RecExpr<L>,
    /// Per-node analysis data; `analysis[i]` corresponds to `rec_expr[i]`.
    pub analysis: Vec<A::Data>,
    /// Per-node subtree sizes; `size[i]` is the number of nodes reachable
    /// from position `i` (counting `i` itself).
    pub size: Vec<usize>,
    /// Per-node costs; `cost[i]` is [`A::cost`][StoAnalysis::cost] for node `i`.
    pub cost: Vec<f64>,
}

impl<L: Language, A: StoAnalysis<L>> State<L, A> {
    /// Build a [`State`] from a [`RecExpr`], computing analysis and sizes
    /// bottom-up in a single pass.
    ///
    /// [`A::modify`][StoAnalysis::modify] is called for each node after the
    /// initial pass, in index order.
    pub fn new(rec_expr: RecExpr<L>) -> Self {
        let n = rec_expr.as_ref().len();
        let mut analysis: Vec<A::Data> = Vec::with_capacity(n);
        let mut size: Vec<usize> = Vec::with_capacity(n);
        let mut cost: Vec<f64> = Vec::with_capacity(n);

        for node in rec_expr.as_ref().iter() {
            let data = A::make(node, &analysis);
            let sz = 1 + node.fold(0usize, |acc, child| acc + size[usize::from(child)]);
            let c = A::cost(node, &analysis, &cost);
            analysis.push(data);
            size.push(sz);
            cost.push(c);
        }

        let mut state = Self {
            rec_expr,
            analysis,
            size,
            cost,
        };

        // Run modify hooks for the initial nodes (new nodes added inside
        // modify will have their own hooks triggered via State::add).
        for i in 0..n {
            A::modify(&mut state, Id::from(i));
        }

        state
    }

    /// The total number of nodes currently in the flat list.
    #[inline]
    pub fn len(&self) -> usize {
        self.rec_expr.as_ref().len()
    }

    /// Whether the state contains no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rec_expr.as_ref().is_empty()
    }

    /// The [`Id`] of the root node (the last element of `rec_expr`).
    ///
    /// # Panics
    ///
    /// Panics if the state is empty.
    #[inline]
    pub fn root(&self) -> Id {
        self.rec_expr.root()
    }

    /// The subtree size of the current root node.
    #[inline]
    pub fn root_size(&self) -> usize {
        self.size[usize::from(self.root())]
    }

    /// Returns `true` when compaction relative to `root` would recover
    /// meaningful memory.
    ///
    /// The heuristic triggers when the total node count exceeds four times the
    /// live subtree size (with a minimum floor of 100 nodes).
    #[inline]
    pub fn should_compact_from(&self, root: Id) -> bool {
        self.len() > std::cmp::max(4 * self.size[usize::from(root)], 100)
    }

    /// Convenience wrapper — equivalent to `should_compact_from(self.root())`.
    #[inline]
    pub fn should_compact(&self) -> bool {
        self.should_compact_from(self.root())
    }

    /// Append a new node to the state, computing its analysis data and subtree
    /// size from its children.
    ///
    /// All children referenced by `node` must already be present in the state.
    /// [`A::modify`][StoAnalysis::modify] is called before returning.
    ///
    /// Returns the [`Id`] of the newly added node.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if any child [`Id`] is out of bounds.
    pub fn add(&mut self, node: L) -> Id {
        debug_assert!(
            node.children().iter().all(|&c| usize::from(c) < self.len()),
            "child id out of bounds"
        );

        let id = self.rec_expr.add(node);
        self.touch(id);
        id
    }

    /// Recompute analysis data and subtree size for the node at `pos`
    pub fn touch(&mut self, pos: Id) {
        let node = &self.rec_expr[pos];
        let data = A::make(&node, &self.analysis);
        let sz = 1 + node.fold(0usize, |acc, child| acc + self.size[usize::from(child)]);
        let c = A::cost(&node, &self.analysis, &self.cost);

        let pos = usize::from(pos);
        if self.analysis.len() <= pos {
            self.analysis.resize(pos + 1, data);
            self.size.resize(pos + 1, sz);
            self.cost.resize(pos + 1, c);
        } else {
            self.analysis[pos] = data;
            self.size[pos] = sz;
            self.cost[pos] = c;
        }
        A::modify(self, Id::from(pos));
    }

    /// Replace every occurrence of `old_pos` as a subterm root with `new_pos`,
    /// within the subtree rooted at `root`.
    fn transplant(&mut self, pos: &mut Id, old_pos: Id, new_pos: Id) {
        if old_pos == *pos {
            *pos = new_pos;
        } else {
            self.transplant_inner(*pos, old_pos, new_pos);
            // self.transplant_inner(new_pos, new_pos, old_pos);
        }
    }

    fn transplant_inner(&mut self, pos: Id, old_pos: Id, new_pos: Id) -> bool {
        if pos == new_pos {
            return false; // Don't recurse into the newly inserted subtree.
        }
        // Clone to release the immutable borrow before any mutable use of self.
        let mut node = self.rec_expr[pos].clone();
        let mut updated = false;
        for child in node.children_mut() {
            if child == &old_pos {
                *child = new_pos;
                updated = true;
            } else {
                updated |= self.transplant_inner(*child, old_pos, new_pos);
            }
        }

        if updated {
            self.rec_expr[pos] = node;
            self.touch(pos);
        }
        updated
    }

    /// Discard all nodes unreachable from **every** root in `roots`, renumber
    /// the remainder, and return the new [`Id`]s corresponding to each input root.
    ///
    /// All nodes reachable from any root are kept; analysis data and sizes are
    /// preserved without recomputation.
    ///
    /// # Panics
    ///
    /// Panics if `roots` is empty or contains an out-of-bounds [`Id`].
    pub fn compact_keeping(&mut self, root: Id) -> Id {
        let n = self.len();
        let reachable = reachable_from(&self.rec_expr, root, n);

        // Collect live indices in ascending order (preserves topological order).
        let order: Vec<usize> = (0..n).filter(|&i| reachable[i]).collect();

        // Build old-index → new-index remapping.
        let mut remap = vec![0u32; n];
        for (new_idx, &old_idx) in order.iter().enumerate() {
            remap[old_idx] = new_idx as u32;
        }

        let old_nodes: Vec<L> = self.rec_expr.as_ref().to_vec();

        let mut new_nodes: Vec<L> = Vec::with_capacity(order.len());
        let mut new_analysis: Vec<A::Data> = Vec::with_capacity(order.len());
        let mut new_size: Vec<usize> = Vec::with_capacity(order.len());
        let mut new_cost: Vec<f64> = Vec::with_capacity(order.len());

        for &old_idx in &order {
            let node = old_nodes[old_idx]
                .clone()
                .map_children(|child| Id::from(remap[usize::from(child)] as usize));
            new_nodes.push(node);
            let mut data = self.analysis[old_idx].clone();
            A::remap_data(&mut data, &remap);
            new_analysis.push(data);
            new_size.push(self.size[old_idx]);
            new_cost.push(self.cost[old_idx]);
        }

        self.rec_expr = RecExpr::from(new_nodes);
        self.analysis = new_analysis;
        self.size = new_size;
        self.cost = new_cost;

        Id::from(remap[usize::from(root)] as usize)
    }

    /// Compact the state keeping only nodes reachable from `self.root()`.
    ///
    /// See [`compact_keeping`][State::compact_keeping] for the general form.
    pub fn compact(&mut self) {
        let root = self.root();
        self.compact_keeping(root);
    }
}

// ─── DFS helper ──────────────────────────────────────────────────────────────

/// Return a boolean mask of length `n` where `mask[i]` is `true` iff node `i`
/// is reachable from at least one node in `roots`.
fn reachable_from<L: Language>(rec_expr: &RecExpr<L>, root: Id, n: usize) -> Vec<bool> {
    let mut visited = vec![false; n];
    let mut stack = Vec::with_capacity(n);
    stack.push(usize::from(root));
    while let Some(i) = stack.pop() {
        if visited[i] {
            continue;
        }
        visited[i] = true;
        for &child in rec_expr[Id::from(i)].children() {
            stack.push(usize::from(child));
        }
    }
    visited
}

// ─── StoSearchMatch ────────────────────────────────────────────────────────

/// A set of matches found by a [`StoSearcher`] at a single position.
///
/// Analogous to [`SearchMatches`][crate::SearchMatches] in the e-graph
/// interface, but refers to a position within a [`RecExpr`] rather than an
/// e-class.
#[derive(Debug, Clone)]
pub struct StoSearchMatch {
    /// Root position in `rec_expr` where the match was found.
    pub pos: Id,
    /// Each substitution maps [`Var`]s to [`Id`]s (positions in `rec_expr`).
    pub substs: Subst,
}

// ─── StoSearcher ─────────────────────────────────────────────────────────────

/// The left-hand side of a [`StoRewrite`].
///
/// A `StoSearcher` inspects a [`State`] and returns positions where a pattern
/// matches, together with substitutions binding pattern variables to sub-term
/// positions.
///
/// Analogous to [`Searcher`][crate::Searcher] in the e-graph interface.
pub trait StoSearcher<L: Language, A: StoAnalysis<L>> {
    /// Search for matches rooted at position `pos`.
    ///
    /// Returns `None` if there is no match at that position.
    fn search_pos(&self, state: &State<L, A>, pos: Id) -> Option<StoSearchMatch>;

    /// Return every [`Var`] that this searcher can bind in a substitution.
    fn vars(&self) -> Vec<Var>;
}

// ─── StoApplier ──────────────────────────────────────────────────────────────

/// The right-hand side of a [`StoRewrite`].
///
/// A `StoApplier` receives a match (position + substitution) and appends a
/// replacement sub-term to the [`State`], returning the [`Id`] of its root.
/// The caller is responsible for splicing it back via [`State::transplant`].
///
/// Analogous to [`Applier`][crate::Applier] in the e-graph interface.
pub trait StoApplier<L: Language, A: StoAnalysis<L>> {
    /// Apply one substitution found at `pos`, extending `state` with new nodes.
    ///
    /// Returns the [`Id`] of the newly created replacement root, or `None` to
    /// skip this match.
    fn apply_one(&self, state: &mut State<L, A>, pos: Id, subst: &Subst) -> Option<Id>;

    /// Return every [`Var`] this applier requires to be bound by the searcher.
    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

// ─── StoConditionalApplier ───────────────────────────────────────────────────

/// A [`StoApplier`] that wraps another applier with a boolean guard.
///
/// If `condition` returns `false` the match is silently skipped — equivalent
/// to [`apply_one`][StoApplier::apply_one] returning `None`.
///
/// Analogous to [`ConditionalApplier`][crate::ConditionalApplier].
pub struct StoConditionalApplier<L: Language, A: StoAnalysis<L>> {
    /// Inner applier to invoke when the condition holds.
    pub applier: Arc<dyn StoApplier<L, A> + Send + Sync>,
    /// Guard predicate.  Receives the *current* state (before the rewrite),
    /// the matched position, and the substitution.  Return `true` to allow
    /// the rewrite, `false` to skip it.
    pub condition: Box<dyn Fn(&State<L, A>, Id, &Subst) -> bool + Send + Sync>,
}

impl<L: Language, A: StoAnalysis<L>> StoApplier<L, A> for StoConditionalApplier<L, A> {
    fn apply_one(&self, state: &mut State<L, A>, pos: Id, subst: &Subst) -> Option<Id> {
        if (self.condition)(state, pos, subst) {
            self.applier.apply_one(state, pos, subst)
        } else {
            None
        }
    }

    fn vars(&self) -> Vec<Var> {
        self.applier.vars()
    }
}

// ─── StoRewrite ──────────────────────────────────────────────────────────────

/// A named rewrite rule for stochastic search.
///
/// Bundles a [`StoSearcher`] and a [`StoApplier`] under a common name and
/// validates at construction time that every applier variable is bound by the
/// searcher.
///
/// Analogous to [`Rewrite`][crate::Rewrite] in the e-graph interface.
pub struct StoRewrite<L, A> {
    /// The name of this rewrite rule.
    pub name: Symbol,
    /// The searcher (left-hand side).
    pub searcher: Arc<dyn StoSearcher<L, A> + Send + Sync>,
    /// The applier (right-hand side).
    pub applier: Arc<dyn StoApplier<L, A> + Send + Sync>,
}

impl<L: Language, A: StoAnalysis<L>> StoRewrite<L, A> {
    /// Create a new [`StoRewrite`], validating variable binding.
    ///
    /// Returns `Err` if any variable referenced by the applier is not bound by
    /// the searcher.
    pub fn new(
        name: impl Into<Symbol>,
        searcher: impl StoSearcher<L, A> + Send + Sync + 'static,
        applier: impl StoApplier<L, A> + Send + Sync + 'static,
    ) -> Result<Self, String> {
        let name = name.into();
        let bound_vars = searcher.vars();
        for v in applier.vars() {
            if !bound_vars.contains(&v) {
                return Err(format!(
                    "StoRewrite `{}`: applier references unbound var `{}`",
                    name, v
                ));
            }
        }
        Ok(Self {
            name,
            searcher: Arc::new(searcher),
            applier: Arc::new(applier),
        })
    }
}

// ─── StoRng ──────────────────────────────────────────────────────────────────

/// Minimal RNG interface for stochastic search.
///
/// Implement this for any RNG you already use.  If you use the `rand` crate,
/// a blanket implementation is two lines:
///
/// ```rust,ignore
/// impl StoRng for rand::rngs::SmallRng {
///     fn gen_float(&mut self) -> f64 { rand::Rng::gen(self) }
///     fn gen_index(&mut self, n: usize) -> usize { rand::Rng::gen_range(self, 0..n) }
/// }
/// ```
pub trait StoRng {
    /// Generate a float uniformly in `[0.0, 1.0)`.
    fn gen_float(&mut self) -> f64;
    /// Generate a `usize` uniformly in `[0, n)`.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    fn gen_index(&mut self, n: usize) -> usize;
}

/// A simple linear congruential generator for tests and examples.
///
/// Not cryptographically secure, but fast and dependency-free.
pub struct SimpleLcg(u64);

impl SimpleLcg {
    /// Create a new LCG from `seed` (must not be zero; `0` is treated as `1`).
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
}

impl StoRng for SimpleLcg {
    fn gen_float(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // 52 bits: max value (2^52-1)*2^-52 = 1-2^-52 is exactly representable
        // in f64, guaranteeing the result stays strictly below 1.0.
        // Using 53 bits risks rounding the maximum to 1.0 under IEEE 754.
        (self.0 >> 12) as f64 * (1.0 / (1u64 << 52) as f64)
    }

    fn gen_index(&mut self, n: usize) -> usize {
        (self.gen_float() * n as f64) as usize
    }
}

// ─── BetaSchedule ────────────────────────────────────────────────────────────

/// A schedule for beta for Metropolis-Hastings / simulated annealing.
pub trait BetaSchedule {
    /// Beta at the given step number (0-indexed).
    fn beta(&self, step: u64) -> f64;
}

/// Normalization hook run after each proposed rewrite application.
///
/// A normalizer receives the proposed full-expression root and may append
/// additional nodes to `state` (for example, by constant folding), returning
/// the root of the normalized expression.
pub trait StoNormalizer<L: Language, A: StoAnalysis<L>> {
    /// Normalize the expression rooted at `root` and return the new root.
    fn normalize(&self, state: &mut State<L, A>, node: L) -> Option<L>;
}

impl<L, A, F> StoNormalizer<L, A> for F
where
    L: Language,
    A: StoAnalysis<L>,
    F: Fn(&mut State<L, A>, L) -> Option<L> + Send + Sync + 'static,
{
    fn normalize(&self, state: &mut State<L, A>, root: L) -> Option<L> {
        self(state, root)
    }
}

struct NoopNormalizer;

impl<L: Language, A: StoAnalysis<L>> StoNormalizer<L, A> for NoopNormalizer {
    fn normalize(&self, _state: &mut State<L, A>, root: L) -> Option<L> {
        None
    }
}

impl<L: Language, A: StoAnalysis<L>> StoNormalizer<L, A>
    for Arc<dyn StoNormalizer<L, A> + Send + Sync>
{
    fn normalize(&self, state: &mut State<L, A>, node: L) -> Option<L> {
        (**self).normalize(state, node)
    }
}

pub struct ConstantBeta(pub f64);

impl BetaSchedule for ConstantBeta {
    fn beta(&self, _step: u64) -> f64 {
        self.0
    }
}

/// Geometric beta: `initial × factor^step`.
///
/// With `factor < 1.0` this converges toward greedy descent.
pub struct GeometricBeta {
    /// Starting beta (at step 0).
    pub initial: f64,
    /// Multiplicative decay per step; factor should be greater than 0
    pub factor: f64,
}

impl BetaSchedule for GeometricBeta {
    fn beta(&self, step: u64) -> f64 {
        self.initial * self.factor.powi(step as i32)
    }
}

pub struct PeriodicBeta;

impl BetaSchedule for PeriodicBeta {
    fn beta(&self, step: u64) -> f64 {
        if step % 1000 < 3 { 0.0 } else { 3.0 }
    }
}

// ─── StoConfig ───────────────────────────────────────────────────────────────

/// Configuration for [`StoRunner::run`].
pub struct StoConfig {
    /// Restart after this many consecutive non-improving iterations.
    /// Default: [`usize::MAX`] (never restart due to stalling).
    pub max_stall: usize,
    /// Maximum number of restarts before stopping.
    /// Default: [`usize::MAX`] (unlimited restarts).
    pub max_restart: usize,
    /// Maximum total iterations.
    /// Default: [`usize::MAX`] (run indefinitely).
    pub max_iter: usize,
    /// Beta schedule for Metropolis-Hastings acceptance.
    /// Default: constant beta of 1.0.
    pub beta_schedule: Box<dyn BetaSchedule>,
}

impl Default for StoConfig {
    fn default() -> Self {
        Self {
            max_stall: usize::MAX,
            max_restart: usize::MAX,
            max_iter: usize::MAX,
            beta_schedule: Box::new(ConstantBeta(1.0)),
        }
    }
}

// ─── StoRunner ────────────────────────────────────────────────────────────────

/// The result of a single Metropolis-Hastings step.
#[derive(Debug, Clone)]
pub struct MhStepResult {
    /// Whether the proposed rewrite improved the current expression.
    pub improved: bool,
    /// The cost of the proposed expression.
    pub proposed_cost: f64,
}

/// A Metropolis-Hastings runner for stochastic term rewriting.
///
/// Maintains a [`State`] and tracks the current and best-ever expressions:
/// - `current_root`: the [`Id`] of the currently accepted expression within
///   `state`.
/// - `best_expr`: a snapshot [`RecExpr`] of the lowest-cost expression seen.
///   It is copied out of the state whenever a new best is found, so compaction
///   of `state` only needs to preserve `current_root`.
///
/// At each step ([`StoRunner::step`]) all matches from every rule are collected
/// and scored via an exponential race that prefers lower-cost proposals while
/// still allowing uphill moves when beta is positive.
///
/// Node costs are read directly from [`State::cost`] — O(1) per step.
pub struct StoRunner<L: Language, A: StoAnalysis<L>> {
    /// The backing state (accumulates all generated nodes until compaction).
    pub state: State<L, A>,
    /// Root of the currently accepted expression within `state`.
    pub current_root: Id,
    /// Cost of the currently accepted expression (`state.cost[current_root]`).
    pub current_cost: f64,
    /// initial expression
    pub initial_expr: RecExpr<L>,
    /// Snapshot of the lowest-cost expression seen so far.
    pub best_expr: RecExpr<L>,
    /// Cost of the best expression seen so far.
    pub best_cost: f64,
    /// Number of [`step`][StoRunner::step] calls made so far.
    pub step_count: u64,
    /// Total number of proposals generated across all steps (for logging / diagnostics).
    pub n_proposed: u64,
    /// Total number of proposals accepted across all steps (for logging / diagnostics).
    pub n_accepted: u64,
    rules: Vec<StoRewrite<L, A>>,
    normalizer: Arc<dyn StoNormalizer<L, A> + Send + Sync>,
}

impl<L: Language + Display, A: StoAnalysis<L>> StoRunner<L, A> {
    /// Create a new [`StoRunner`] from an initial state and rules.
    pub fn new(initial_expr: RecExpr<L>, rules: Vec<StoRewrite<L, A>>) -> Self {
        let state = State::new(initial_expr.clone());
        let current_root = state.root();
        let current_cost = state.cost[usize::from(current_root)];
        let best_expr = state.rec_expr.extract(current_root);
        Self {
            state,
            current_root,
            current_cost,
            initial_expr,
            best_expr,
            best_cost: current_cost,
            step_count: 0,
            n_proposed: 0,
            n_accepted: 0,
            rules,
            normalizer: Arc::new(NoopNormalizer),
        }
    }

    /// Set a post-rewrite normalizer invoked on every proposed expression.
    ///
    /// The normalizer runs after a rewrite is transplanted into the full term
    /// and before MH acceptance is evaluated.
    pub fn with_normalizer(
        mut self,
        normalizer: impl StoNormalizer<L, A> + Send + Sync + 'static,
    ) -> Self {
        self.normalizer = Arc::new(normalizer);
        self
    }

    /// Normalize the subtree rooted at `pos` in-place, updating analysis and
    /// cost for every touched node.
    pub fn normalize(&mut self, pos: Id) {
        let mut normalizer = Arc::clone(&self.normalizer);
        if Self::normalize_inner(&mut self.state, &mut normalizer, pos) {
            self.state.touch(pos);
        }
    }

    fn normalize_inner(
        state: &mut State<L, A>,
        normalizer: &mut (impl StoNormalizer<L, A> + Send + Sync),
        pos: Id,
    ) -> bool {
        // eprintln!(
        //     "Normalizing at pos {}: {:?} (size: {})",
        //     pos,
        //     state.rec_expr[pos],
        //     state.size[usize::from(pos)]
        // );
        let node = state.rec_expr[pos].clone();
        let mut updated = false;
        for i in node.children() {
            updated |= Self::normalize_inner(state, normalizer, *i);
        }

        if let Some(folded) = normalizer.normalize(state, node) {
            state.rec_expr[pos] = folded;
            updated = true;
        }

        if updated {
            state.touch(pos);
        }

        updated
    }

    /// Perform one Metropolis-Hastings step.
    pub fn step<T: BetaSchedule + ?Sized>(&mut self, schedule: &T, rng: &mut impl StoRng) -> MhStepResult {
        assert!(!self.rules.is_empty());

        let beta = schedule.beta(self.step_count);
        self.step_count += 1;

        let mut sample_exp1 = || {
            let u = rng.gen_float();
            let u = u.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
            -(-u.ln()).ln()
        };

        // Incumbent starts as the current state.
        // sample_exp1();
        let mut selected_nonce = sample_exp1();
        let mut selected_cost = self.current_cost;
        let mut winner: Option<(Id, Id)> = None; // (old_subterm_pos, new_subterm)

        // Enumerate all proposals without storing them: maintain only the
        // current race winner.

        let reachable = reachable_from(&self.state.rec_expr, self.current_root, self.state.len());
        let reachable = reachable
            .into_iter()
            .enumerate()
            .filter(|&(_, reachable)| reachable)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let search_rooted =
            |state: &State<L, A>, searcher: &Arc<dyn StoSearcher<L, A> + Send + Sync + 'static>| {
                reachable
                    .iter()
                    .filter_map(|i| searcher.search_pos(state, Id::from(*i)))
                    .collect::<Vec<_>>()
            };
        let rules = mem::take(&mut self.rules);
        for rule in &rules {
            for m in search_rooted(&self.state, &rule.searcher) {
                let Some(new_subterm) = rule.applier.apply_one(&mut self.state, m.pos, &m.substs)
                else {
                    continue;
                };
                self.n_proposed += 1;
                self.normalize(new_subterm);

                let current_subterm_cost = self.state.cost[usize::from(m.pos)];
                let proposed_subterm_cost = self.state.cost[usize::from(new_subterm)];

                let weight = -beta * (proposed_subterm_cost - current_subterm_cost);
                let nonce = weight + sample_exp1();
                if nonce > selected_nonce {
                    selected_nonce = nonce;
                    winner = Some((m.pos, new_subterm));
                }
            }
        }
        self.rules = rules;

        // Apply the winning proposal exactly once, after selection is complete.
        if let Some((old_pos, new_subterm)) = winner {
            self.state
                .transplant(&mut self.current_root, old_pos, new_subterm);
            self.normalize(self.current_root);
            selected_cost = self.state.cost[usize::from(self.current_root)];
            self.n_accepted += 1;
        }

        let improved = selected_cost < self.current_cost;
        if improved {
            self.current_cost = selected_cost;

            if selected_cost < self.best_cost {
                self.best_expr = self.state.rec_expr.extract(self.current_root);
                self.best_cost = selected_cost;
            }
        }

        // ── 7. Compact if beneficial ──────────────────────────────────────────
        if self.state.should_compact_from(self.current_root) {
            let new_id = self.state.compact_keeping(self.current_root);
            self.current_root = new_id;
        }

        MhStepResult {
            improved: improved,
            proposed_cost: selected_cost,
        }
    }

    /// Run the MH chain according to the given [`StoConfig`].
    ///
    /// Stops when `max_iter` iterations have been run, when `max_restart`
    /// restarts have occurred, or when `max_stall` consecutive non-improving
    /// iterations are reached and no further restarts are allowed.
    pub fn run(&mut self, config: StoConfig, rng: &mut impl StoRng) {
        let mut stall = 0usize;
        let mut restarts = 0usize;
        for _ in 0..config.max_iter {
            let result = self.step(config.beta_schedule.as_ref(), rng);
            stall += !result.improved as usize;
            if stall >= config.max_stall {
                if restarts >= config.max_restart {
                    break;
                }
                self.state = State::new(self.initial_expr.clone());
                self.current_root = self.state.root();
                self.current_cost = self.state.cost[usize::from(self.current_root)];
                stall = 0;
                restarts += 1;
            }
        }
        println!(
            "MH finished: best cost = {}, # proposed = {}, # accepted = {}",
            self.best_cost, self.n_proposed, self.n_accepted,
        );
    }

    /// Extract the current (most recently accepted) expression as a standalone [`RecExpr`].
    pub fn current_expr(&self) -> RecExpr<L> {
        self.state.rec_expr.extract(self.current_root)
    }
}
