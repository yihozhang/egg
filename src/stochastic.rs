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
//! [`MhRunner`] implements Metropolis-Hastings over the space of terms.
//! At each step it randomly picks a rule, randomly picks a match, applies the
//! rewrite (via [`State::transplant`]), and accepts or rejects the result
//! according to the standard MH criterion using [`StoAnalysis::cost`] and a
//! [`TempSchedule`].
//!
//! # Substitutions
//!
//! Pattern variables are bound to [`Id`]s that serve as *positions* in
//! `rec_expr` — the same role [`Id`]s play as e-class identifiers in the
//! e-graph interface.  The existing [`Subst`] type is reused directly.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

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

        let data = A::make(&node, &self.analysis);
        let sz = 1 + node.fold(0usize, |acc, child| acc + self.size[usize::from(child)]);
        let c = A::cost(&node, &self.analysis, &self.cost);

        let id = self.rec_expr.add(node);
        self.analysis.push(data);
        self.size.push(sz);
        self.cost.push(c);

        A::modify(self, id);
        id
    }

    /// Replace every occurrence of `old_pos` as a subterm root with `new_pos`,
    /// within the subtree rooted at `root`, appending rebuilt ancestor nodes
    /// as needed.
    ///
    /// Nodes whose subtrees do not contain `old_pos` are reused unchanged.
    /// Rebuilds are memoized so each position is visited at most once.
    ///
    /// Returns the new root of the modified subtree.
    pub fn transplant(&mut self, root: Id, old_pos: Id, new_pos: Id) -> Id {
        if root == old_pos {
            return new_pos;
        }
        let mut memo = HashMap::new();
        self.do_transplant(root, old_pos, new_pos, &mut memo)
    }

    fn do_transplant(
        &mut self,
        pos: Id,
        old_pos: Id,
        new_pos: Id,
        memo: &mut HashMap<Id, Id>,
    ) -> Id {
        if pos == old_pos {
            return new_pos;
        }
        if let Some(&cached) = memo.get(&pos) {
            return cached;
        }
        // Clone to release the immutable borrow before any mutable use of self.
        let node = self.rec_expr[pos].clone();
        let new_children: Vec<Id> = node
            .children()
            .iter()
            .map(|&child| self.do_transplant(child, old_pos, new_pos, memo))
            .collect();

        let result = if node.children() == new_children.as_slice() {
            pos
        } else {
            let mut iter = new_children.into_iter();
            let new_node = node.map_children(|_| iter.next().unwrap());
            self.add(new_node)
        };
        memo.insert(pos, result);
        result
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
    pub fn compact_keeping(&mut self, roots: &[Id]) -> Vec<Id> {
        let n = self.len();
        let reachable = reachable_from(&self.rec_expr, roots, n);

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
            new_analysis.push(self.analysis[old_idx].clone());
            new_size.push(self.size[old_idx]);
            new_cost.push(self.cost[old_idx]);
        }

        self.rec_expr = RecExpr::from(new_nodes);
        self.analysis = new_analysis;
        self.size = new_size;
        self.cost = new_cost;

        roots
            .iter()
            .map(|&r| Id::from(remap[usize::from(r)] as usize))
            .collect()
    }

    /// Compact the state keeping only nodes reachable from `self.root()`.
    ///
    /// See [`compact_keeping`][State::compact_keeping] for the general form.
    pub fn compact(&mut self) {
        let root = self.root();
        self.compact_keeping(&[root]);
    }
}

// ─── DFS helper ──────────────────────────────────────────────────────────────

/// Return a boolean mask of length `n` where `mask[i]` is `true` iff node `i`
/// is reachable from at least one node in `roots`.
///
/// Callers can pass `n = usize::from(root) + 1` when there is a single root,
/// since children always have smaller indices in a [`RecExpr`].
fn reachable_from<L: Language>(rec_expr: &RecExpr<L>, roots: &[Id], n: usize) -> Vec<bool> {
    let mut visited = vec![false; n];
    for &root in roots {
        let mut stack = vec![usize::from(root)];
        while let Some(i) = stack.pop() {
            if visited[i] {
                continue;
            }
            visited[i] = true;
            for &child in rec_expr[Id::from(i)].children() {
                stack.push(usize::from(child));
            }
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

    /// Search only the subtree reachable from `root`.
    ///
    /// Used by [`MhRunner`] so that proposals are confined to the currently
    /// accepted expression.  The default does a DFS from `root` and calls
    /// [`search_pos`][Self::search_pos] at each reachable position.
    fn search_rooted(&self, state: &State<L, A>, root: Id) -> Vec<StoSearchMatch> {
        // Children always have smaller indices than their parent in a RecExpr,
        // so only indices 0..=root can be in the subtree.
        let n = usize::from(root) + 1;
        reachable_from(&state.rec_expr, &[root], n)
            .into_iter()
            .enumerate()
            .filter(|&(_, reachable)| reachable)
            .filter_map(|(i, _)| self.search_pos(state, Id::from(i)))
            .collect()
    }

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

// ─── TempSchedule ────────────────────────────────────────────────────────────

/// A temperature schedule for Metropolis-Hastings / simulated annealing.
///
/// The temperature controls acceptance of cost-increasing proposals: at high
/// temperature almost anything is accepted; at zero only improvements are
/// accepted (greedy descent).
pub trait TempSchedule {
    /// Temperature at the given step number (0-indexed).
    fn temperature(&self, step: u64) -> f64;
}

/// Constant temperature — no annealing.
pub struct ConstantTemp(pub f64);

impl TempSchedule for ConstantTemp {
    fn temperature(&self, _step: u64) -> f64 {
        self.0
    }
}

/// Geometrically-decaying temperature: `initial × factor^step`.
///
/// With `factor < 1.0` this converges toward greedy descent.
pub struct GeometricTemp {
    /// Starting temperature (at step 0).
    pub initial: f64,
    /// Multiplicative decay per step; use a value in `(0, 1)` for annealing.
    pub factor: f64,
}

impl TempSchedule for GeometricTemp {
    fn temperature(&self, step: u64) -> f64 {
        if step <= i32::MAX as u64 {
            self.initial * self.factor.powi(step as i32)
        } else {
            self.initial * self.factor.powf(step as f64)
        }
    }
}

// ─── MhRunner ────────────────────────────────────────────────────────────────

/// The result of a single Metropolis-Hastings step.
#[derive(Debug, Clone)]
pub struct MhStepResult {
    /// Whether the proposed rewrite was accepted.
    pub accepted: bool,
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
/// At each step ([`MhRunner::step`]) all matches from every rule are collected
/// and one is sampled uniformly.  The rewrite is applied and accepted/rejected
/// via:
///
/// ```text
/// accept if new_cost ≤ current_cost,
/// else accept with probability exp((current_cost − new_cost) / temperature).
/// ```
///
/// Node costs are read directly from [`State::cost`] — O(1) per step.
pub struct MhRunner<L: Language, A: StoAnalysis<L>> {
    /// The backing state (accumulates all generated nodes until compaction).
    pub state: State<L, A>,
    /// Root of the currently accepted expression within `state`.
    pub current_root: Id,
    /// Cost of the currently accepted expression (`state.cost[current_root]`).
    pub current_cost: f64,
    /// Snapshot of the lowest-cost expression seen so far.
    pub best_expr: RecExpr<L>,
    /// Cost of the best expression seen so far.
    pub best_cost: f64,
    /// Number of [`step`][MhRunner::step] calls made so far.
    pub step_count: u64,
    rules: Vec<StoRewrite<L, A>>,
}

impl<L: Language, A: StoAnalysis<L>> MhRunner<L, A> {
    /// Create a new [`MhRunner`] from an initial state and rules.
    pub fn new(state: State<L, A>, rules: Vec<StoRewrite<L, A>>) -> Self {
        let current_root = state.root();
        let current_cost = state.cost[usize::from(current_root)];
        let best_expr = state.rec_expr.extract(current_root);
        Self {
            state,
            current_root,
            current_cost,
            best_expr,
            best_cost: current_cost,
            step_count: 0,
            rules,
        }
    }

    /// Perform one Metropolis-Hastings step.
    ///
    /// 1. Collect all matches from every rule within the subtree rooted at
    ///    `current_root`.
    /// 2. Sample one match uniformly at random.
    /// 3. Apply the rewrite to produce a new subterm root.
    /// 4. [`transplant`][State::transplant] the new subterm into the full
    ///    expression.
    /// 5. Accept with probability `min(1, exp((current_cost − new_cost) / T))`.
    /// 6. If a new best is found, snapshot `best_expr`.
    /// 7. Compact the state if the dead-to-live ratio is high.
    ///
    /// Returns `None` if there are no rules; otherwise always returns
    /// `Some(MhStepResult)` — with `accepted = false` when no match was found
    /// or the proposal was rejected.
    pub fn step<T: TempSchedule>(&mut self, schedule: &T, rng: &mut impl StoRng) -> Option<MhStepResult> {
        if self.rules.is_empty() {
            return None;
        }

        let temp = schedule.temperature(self.step_count);
        self.step_count += 1;

        // ── 1–2. Reservoir-sample one (applier, match) pair uniformly ─────────
        // Scanning all rules and positions with a size-1 reservoir avoids
        // building a Vec of all matches, reducing allocations on the hot path.
        let mut chosen: Option<(Arc<dyn StoApplier<L, A> + Send + Sync>, StoSearchMatch)> = None;
        let mut total = 0usize;
        for rule in &self.rules {
            for m in rule.searcher.search_rooted(&self.state, self.current_root) {
                total += 1;
                if rng.gen_index(total) == 0 {
                    chosen = Some((Arc::clone(&rule.applier), m));
                }
            }
        }

        let (applier, chosen_match) = match chosen {
            None => {
                return Some(MhStepResult {
                    accepted: false,
                    proposed_cost: self.current_cost,
                });
            }
            Some(pair) => pair,
        };

        // ── 3. Apply the rewrite ──────────────────────────────────────────────
        let new_subterm =
            match applier.apply_one(&mut self.state, chosen_match.pos, &chosen_match.substs) {
                Some(id) => id,
                None => {
                    return Some(MhStepResult {
                        accepted: false,
                        proposed_cost: self.current_cost,
                    });
                }
            };

        // ── 4. Transplant into the full expression ────────────────────────────
        let proposed_root = self
            .state
            .transplant(self.current_root, chosen_match.pos, new_subterm);

        // ── 5. MH acceptance ──────────────────────────────────────────────────
        let proposed_cost = self.state.cost[usize::from(proposed_root)];

        let accepted = if proposed_cost <= self.current_cost {
            true
        } else if temp <= 0.0 {
            false
        } else {
            let delta = proposed_cost - self.current_cost;
            rng.gen_float() < (-delta / temp).exp()
        };

        if accepted {
            self.current_root = proposed_root;
            self.current_cost = proposed_cost;

            if proposed_cost < self.best_cost {
                self.best_expr = self.state.rec_expr.extract(proposed_root);
                self.best_cost = proposed_cost;
            }
        }

        if self.state.should_compact_from(self.current_root) {
            let new_ids = self.state.compact_keeping(&[self.current_root]);
            self.current_root = new_ids[0];
        }

        Some(MhStepResult {
            accepted,
            proposed_cost,
        })
    }

    /// Run the MH chain for exactly `n_steps` steps.
    ///
    /// Returns early if there are no rules.
    pub fn run<T: TempSchedule>(&mut self, n_steps: u64, schedule: &T, rng: &mut impl StoRng) {
        for _ in 0..n_steps {
            if self.step(schedule, rng).is_none() {
                break;
            }
        }
    }

    /// Extract the current (most recently accepted) expression as a standalone [`RecExpr`].
    pub fn current_expr(&self) -> RecExpr<L> {
        self.state.rec_expr.extract(self.current_root)
    }
}
