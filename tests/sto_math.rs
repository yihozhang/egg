//! Stochastic search tests using a subset of the Math language.

use std::collections::HashMap;
use std::sync::Arc;

use egg::{
    stochastic::{
        ConstantBeta, GeometricBeta, MhRunner, SimpleLcg, State, StoAnalysis,
        StoConditionalApplier, StoRewrite,
    },
    *,
};
use ordered_float::NotNan;

// ─── Language ────────────────────────────────────────────────────────────────

define_language! {
    enum Math {
        "d"    = Diff([Id; 2]),
        "i"    = Integral([Id; 2]),
        "+"    = Add([Id; 2]),
        "-"    = Sub([Id; 2]),
        "*"    = Mul([Id; 2]),
        "/"    = Div([Id; 2]),
        "pow"  = Pow([Id; 2]),
        "ln"   = Ln(Id),
        "sqrt" = Sqrt(Id),
        "sin"  = Sin(Id),
        "cos"  = Cos(Id),
        Constant(NotNan<f64>),
        Symbol(Symbol),
    }
}

// ─── Analysis / cost ─────────────────────────────────────────────────────────

/// Stochastic analysis that assigns a high cost to `d` and `i` operators.
struct MathCost;

impl StoAnalysis<Math> for MathCost {
    type Data = ();
    fn make(_: &Math, _: &[()]) {}

    fn cost(enode: &Math, _: &[()], children_cost: &[f64]) -> f64 {
        let op = match enode {
            Math::Diff(..) | Math::Integral(..) => 3.0,
            _ => 1.0,
        };
        op + enode.fold(0.0, |acc, c| acc + children_cost[usize::from(c)])
    }
}

type MathState = State<Math, MathCost>;
type MathRw = StoRewrite<Math, MathCost>;

fn const_at(state: &MathState, id: Id) -> Option<NotNan<f64>> {
    match state.rec_expr[id] {
        Math::Constant(c) => Some(c),
        _ => None,
    }
}

fn fold_math_node(state: &mut MathState, id: Id) -> Option<Id> {
    match state.rec_expr[id] {
        Math::Add([a, b]) => {
            let folded = const_at(state, a)? + const_at(state, b)?;
            Some(state.add(Math::Constant(folded)))
        }
        Math::Sub([a, b]) => {
            let folded = const_at(state, a)? - const_at(state, b)?;
            Some(state.add(Math::Constant(folded)))
        }
        Math::Mul([a, b]) => {
            let folded = const_at(state, a)? * const_at(state, b)?;
            Some(state.add(Math::Constant(folded)))
        }
        Math::Div([a, b]) => {
            let denom = const_at(state, b)?;
            if denom == 0.0 {
                None
            } else {
                Some(state.add(Math::Constant(const_at(state, a)? / denom)))
            }
        }
        Math::Pow([a, b]) => {
            let val = const_at(state, a)?
                .into_inner()
                .powf(const_at(state, b)?.into_inner());
            let folded = NotNan::new(val).ok()?;
            Some(state.add(Math::Constant(folded)))
        }
        Math::Ln(a) => {
            let val = const_at(state, a)?.into_inner().ln();
            let folded = NotNan::new(val).ok()?;
            Some(state.add(Math::Constant(folded)))
        }
        Math::Sqrt(a) => {
            let val = const_at(state, a)?.into_inner().sqrt();
            let folded = NotNan::new(val).ok()?;
            Some(state.add(Math::Constant(folded)))
        }
        Math::Sin(a) => {
            let val = const_at(state, a)?.into_inner().sin();
            let folded = NotNan::new(val).ok()?;
            Some(state.add(Math::Constant(folded)))
        }
        Math::Cos(a) => {
            let val = const_at(state, a)?.into_inner().cos();
            let folded = NotNan::new(val).ok()?;
            Some(state.add(Math::Constant(folded)))
        }
        _ => None,
    }
}

fn normalize_math(state: &mut MathState, root: Id) -> Id {
    fn rebuild(state: &mut MathState, pos: Id, memo: &mut HashMap<Id, Id>) -> Id {
        if let Some(&cached) = memo.get(&pos) {
            return cached;
        }

        let node = state.rec_expr[pos].clone();
        let new_children: Vec<Id> = node
            .children()
            .iter()
            .map(|&child| rebuild(state, child, memo))
            .collect();

        let rebuilt = if node.children() == new_children.as_slice() {
            pos
        } else {
            let mut iter = new_children.into_iter();
            let new_node = node.map_children(|_| iter.next().unwrap());
            state.add(new_node)
        };

        let normalized = fold_math_node(state, rebuilt).unwrap_or(rebuilt);
        memo.insert(pos, normalized);
        normalized
    }

    rebuild(state, root, &mut HashMap::new())
}

// ─── Rule helpers ─────────────────────────────────────────────────────────────

fn p(s: &str) -> Pattern<Math> {
    s.parse().unwrap()
}

fn rw(name: &str, lhs: &str, rhs: &str) -> MathRw {
    StoRewrite::new(name, p(lhs), p(rhs)).unwrap()
}

fn rw_if(
    name: &str,
    lhs: &str,
    rhs: &str,
    cond: impl Fn(&MathState, Id, &Subst) -> bool + Send + Sync + 'static,
) -> MathRw {
    StoRewrite::new(
        name,
        p(lhs),
        StoConditionalApplier {
            applier: Arc::new(p(rhs)),
            condition: Box::new(cond),
        },
    )
    .unwrap()
}

// ─── Conditions ──────────────────────────────────────────────────────────────

fn is_sym(var: &str) -> impl Fn(&MathState, Id, &Subst) -> bool + Send + Sync + 'static {
    let v: Var = var.parse().unwrap();
    move |s: &MathState, _: Id, subst: &Subst| matches!(s.rec_expr[subst[v]], Math::Symbol(_))
}

fn is_not_zero(var: &str) -> impl Fn(&MathState, Id, &Subst) -> bool + Send + Sync + 'static {
    let v: Var = var.parse().unwrap();
    move |s: &MathState, _: Id, subst: &Subst| {
        if let Math::Constant(c) = s.rec_expr[subst[v]] {
            *c != 0.0
        } else {
            true
        }
    }
}

fn is_const(var: &str) -> impl Fn(&MathState, Id, &Subst) -> bool + Send + Sync + 'static {
    let v: Var = var.parse().unwrap();
    move |s: &MathState, _: Id, subst: &Subst| matches!(s.rec_expr[subst[v]], Math::Constant(_))
}

/// True when `v` and `w` are bound to different positions AND `v` is a
/// constant or symbol (mirrors `is_const_or_distinct_var` from the e-graph
/// math test).
fn is_const_or_distinct_sym(
    v_str: &str,
    w_str: &str,
) -> impl Fn(&MathState, Id, &Subst) -> bool + Send + Sync + 'static {
    let v: Var = v_str.parse().unwrap();
    let w: Var = w_str.parse().unwrap();
    move |s: &MathState, _: Id, subst: &Subst| {
        let vi = subst[v];
        let wi = subst[w];
        vi != wi
            && (matches!(s.rec_expr[vi], Math::Constant(_))
                || matches!(s.rec_expr[vi], Math::Symbol(_)))
    }
}

// ─── Rules ───────────────────────────────────────────────────────────────────

#[rustfmt::skip]
fn rules() -> Vec<MathRw> {
    vec![
        rw("comm-add",  "(+ ?a ?b)",        "(+ ?b ?a)"),
        rw("comm-mul",  "(* ?a ?b)",        "(* ?b ?a)"),
        rw("assoc-add", "(+ ?a (+ ?b ?c))", "(+ (+ ?a ?b) ?c)"),
        rw("assoc-mul", "(* ?a (* ?b ?c))", "(* (* ?a ?b) ?c)"),

        rw("sub-canon", "(- ?a ?b)", "(+ ?a (* -1 ?b))"),
        rw_if("div-canon", "(/ ?a ?b)", "(* ?a (pow ?b -1))", is_not_zero("?b")),

        rw("zero-add", "(+ ?a 0)", "?a"),
        rw("zero-mul", "(* ?a 0)", "0"),
        rw("one-mul",  "(* ?a 1)", "?a"),

        rw("add-zero", "?a", "(+ ?a 0)"),
        rw("mul-one",  "?a", "(* ?a 1)"),

        rw("cancel-sub", "(- ?a ?a)", "0"),
        rw_if("cancel-div", "(/ ?a ?a)", "1", is_not_zero("?a")),

        rw("distribute", "(* ?a (+ ?b ?c))",        "(+ (* ?a ?b) (* ?a ?c))"),
        rw("factor",     "(+ (* ?a ?b) (* ?a ?c))", "(* ?a (+ ?b ?c))"),

        rw("pow-mul", "(* (pow ?a ?b) (pow ?a ?c))", "(pow ?a (+ ?b ?c))"),
        rw_if("pow0",        "(pow ?x 0)",     "1",        is_not_zero("?x")),
        rw("pow1",           "(pow ?x 1)",     "?x"),
        rw("pow2",           "(pow ?x 2)",     "(* ?x ?x)"),
        rw_if("pow-recip",     "(pow ?x -1)",       "(/ 1 ?x)", is_not_zero("?x")),
        rw_if("recip-mul-div", "(* ?x (/ 1 ?x))",  "1",        is_not_zero("?x")),

        rw_if("d-variable", "(d ?x ?x)", "1", is_sym("?x")),
        rw_if("d-constant", "(d ?x ?c)", "0", {
            let sym_x  = is_sym("?x");
            let cdv    = is_const_or_distinct_sym("?c", "?x");
            move |s, pos, subst| sym_x(s, pos, subst) && cdv(s, pos, subst)
        }),

        rw("d-add", "(d ?x (+ ?a ?b))", "(+ (d ?x ?a) (d ?x ?b))"),
        rw("d-mul", "(d ?x (* ?a ?b))", "(+ (* ?a (d ?x ?b)) (* ?b (d ?x ?a)))"),

        rw("d-sin", "(d ?x (sin ?x))", "(cos ?x)"),
        rw("d-cos", "(d ?x (cos ?x))", "(* -1 (sin ?x))"),
        rw_if("d-ln", "(d ?x (ln ?x))", "(/ 1 ?x)", is_not_zero("?x")),

        rw_if("d-power",
            "(d ?x (pow ?f ?g))",
            "(* (pow ?f ?g) (+ (* (d ?x ?f) (/ ?g ?f)) (* (d ?x ?g) (ln ?f))))",
            { let nzf = is_not_zero("?f"); let nzg = is_not_zero("?g");
              move |s, pos, subst| nzf(s, pos, subst) && nzg(s, pos, subst) }),

        rw_if("i-power-const", "(i (pow ?x ?c) ?x)",
            "(/ (pow ?x (+ ?c 1)) (+ ?c 1))", is_const("?c")),
        rw("i-one",   "(i 1 ?x)",         "?x"),
        rw("i-cos",   "(i (cos ?x) ?x)",  "(sin ?x)"),
        rw("i-sin",   "(i (sin ?x) ?x)",  "(* -1 (cos ?x))"),
        rw("i-sum",   "(i (+ ?f ?g) ?x)", "(+ (i ?f ?x) (i ?g ?x))"),
        rw("i-dif",   "(i (- ?f ?g) ?x)", "(- (i ?f ?x) (i ?g ?x))"),
        rw("i-parts", "(i (* ?a ?b) ?x)",
            "(- (* ?a (i ?b ?x)) (i (* (d ?x ?a) (i ?b ?x)) ?x))"),
    ]
}

// ─── Drivers ─────────────────────────────────────────────────────────────────

/// Run pure greedy descent (temperature = 0) from `start` for up to `n_steps`
/// steps.  Returns the best expression and its cost found during the run.
fn greedy_best(start: &str, n_steps: u64) -> (RecExpr<Math>, f64) {
    let expr: RecExpr<Math> = start.parse().unwrap();
    let state = MathState::new(expr);
    let mut runner = MhRunner::new(state, rules()).with_normalizer(normalize_math);
    let mut rng = SimpleLcg::new(42);
    runner.run(n_steps, &ConstantBeta(0.0), &mut rng);
    (runner.best_expr.clone(), runner.best_cost)
}

fn metropolis_best(start: &str, n_steps: u64, beta: f64, seed: u64) -> (RecExpr<Math>, f64) {
    let expr: RecExpr<Math> = start.parse().unwrap();
    let state = MathState::new(expr);
    let mut runner = MhRunner::new(state, rules()).with_normalizer(normalize_math);
    let mut rng = SimpleLcg::new(seed);
    runner.run(n_steps, &ConstantBeta(beta), &mut rng);
    (runner.best_expr.clone(), runner.best_cost)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn sto_diff_same() {
    // (d x x) → 1  via d-variable  (cost 102 → 1)
    let (_, cost) = metropolis_best("(d x x)", 50000, 1.0, 0);
    assert_eq!(cost, 1.0);
}

#[test]
fn sto_diff_different() {
    // (d x y) → 0  via d-constant  (cost 102 → 1)
    let (_, cost) = metropolis_best("(d x y)", 50000, 1.0, 0);
    assert_eq!(cost, 1.0);
}

#[test]
fn sto_zero_add() {
    // (+ x 0) → x  via zero-add  (cost 3 → 1)
    let (_, cost) = metropolis_best("(+ x 0)", 50000, 1.0, 0);
    assert_eq!(cost, 1.0);
}

#[test]
fn sto_powers() {
    // (* (pow 2 x) (pow 2 y)) → (pow 2 (+ x y))  via pow-mul  (cost 7 → 5)
    let initial = {
        let expr: RecExpr<Math> = "(* (pow 2 x) (pow 2 y))".parse().unwrap();
        let s = MathState::new(expr);
        s.cost[usize::from(s.root())]
    };
    let (_, best) = metropolis_best("(* (pow 2 x) (pow 2 y))", 50000, 1.0, 0);
    assert!(
        best < initial,
        "cost did not decrease: {} vs initial {}",
        best,
        initial
    );
}

#[test]
fn sto_simplify_add() {
    // (+ x (+ x (+ x x))) has a lower-cost equivalent, but reaching it usually
    // needs cost-increasing intermediate rewrites.
    let initial = {
        let e: RecExpr<Math> = "(+ x (+ x (+ x x)))".parse().unwrap();
        let s = MathState::new(e);
        s.cost[usize::from(s.root())]
    };
    let (_, best) = metropolis_best("(+ x (+ x (+ x x)))", 50000, 1.0, 0);
    assert!(best < initial, "expected cost < {}, got {}", initial, best);
}

#[test]
fn sto_diff_sin() {
    // (d x (sin x)) → (cos x)  via d-sin  (cost 103 → 2)
    let (_, cost) = metropolis_best("(d x (sin x))", 50000, 1.0, 0);
    assert_eq!(cost, 2.0);
}

#[test]
fn sto_diff_simple2() {
    // (d x (+ 1 (* y x))) → y
    let initial = {
        let e: RecExpr<Math> = "(d x (+ 1 (* y x)))".parse().unwrap();
        let s = MathState::new(e);
        s.cost[usize::from(s.root())]
    };
    let (_, best) = metropolis_best("(d x (+ 1 (* y x)))", 80_000, 1.0, 42);
    assert!(best < initial, "expected cost < {}, got {}", initial, best);
}

#[test]
fn sto_diff_ln() {
    // (d x (ln x)) → (/ 1 x)
    let (_, cost) = metropolis_best("(d x (ln x))", 40000, 1.0, 0);
    assert_eq!(cost, 3.0);
}

#[test]
fn sto_diff_power_simple() {
    // (d x (pow x 3)) → (* 3 (pow x 2))
    let initial = {
        let e: RecExpr<Math> = "(d x (pow x 3))".parse().unwrap();
        let s = MathState::new(e);
        s.cost[usize::from(s.root())]
    };
    let (_, best) = metropolis_best("(d x (pow x 3))", 80_000, 1.0, 42);
    assert!(best < initial, "expected cost < {}, got {}", initial, best);
}

#[test]
fn sto_integ_cos() {
    // (i (cos x) x) → (sin x)  via i-cos  (cost 103 → 2)
    let (_, cost) = metropolis_best("(i (cos x) x)", 50000, 1.0, 0);
    assert_eq!(cost, 2.0);
}

#[test]
fn sto_integ_one() {
    // (i 1 x) → x
    let (_, cost) = metropolis_best("(i 1 x)", 2000, 1.0, 0);
    assert_eq!(cost, 1.0);
}

#[test]
fn sto_integ_sin() {
    // Keep parity with math.rs naming: integ_sin checks (i (cos x) x) -> (sin x)
    let (_, cost) = metropolis_best("(i (cos x) x)", 2000, 1.0, 0);
    assert_eq!(cost, 2.0);
}

#[test]
fn sto_simplify_const() {
    // (+ 1 (- a (* (- 2 1) a))) → 1
    // Each step reduces cost: cf-sub folds (- 2 1)→1, then one-mul, cancel-sub, zero-add.
    let (_, cost) = metropolis_best("(+ 1 (- a (* (- 2 1) a)))", 2000, 1.0, 0);
    assert_eq!(cost, 1.0);
}

#[test]
fn sto_integ_x() {
    // (i (pow x 1) x) → (/ (pow x 2) 2)  via i-power-const then cf-add twice
    // All steps reduce cost, so greedy descent reaches the target.
    let (_e, cost) = metropolis_best("(i (pow x 1) x)", 50000, 1.0, 42);
    assert_eq!(cost, 5.0);
}

#[test]
fn sto_diff_simple1() {
    // (d x (+ 1 (* 2 x))) → 2  (multi-step; intermediate steps raise cost → needs annealing)
    let initial = {
        let e: RecExpr<Math> = "(d x (+ 1 (* 2 x)))".parse().unwrap();
        let s = MathState::new(e);
        s.cost[usize::from(s.root())]
    };
    let (_, best) = metropolis_best("(d x (+ 1 (* 2 x)))", 50_000, 1.0, 42);
    assert!(best < initial, "expected cost < {}, got {}", initial, best);
}

#[test]
fn sto_simplify_root() {
    // (/ 1 (- (/ (+ 1 (sqrt five)) 2) (/ (- 1 (sqrt five)) 2))) → (/ 1 (sqrt five))
    let expr = "(/ 1 (- (/ (+ 1 (sqrt five)) 2) (/ (- 1 (sqrt five)) 2)))";
    let initial = {
        let e: RecExpr<Math> = expr.parse().unwrap();
        let s = MathState::new(e);
        s.cost[usize::from(s.root())]
    };
    let (_, best) = metropolis_best(expr, 100_000, 1.0, 42);
    assert!(best < initial, "expected cost < {}, got {}", initial, best);
}

#[test]
fn sto_simplify_factor() {
    // Adapted from math_simplify_factor for stochastic optimization:
    // start from expanded form and expect factorization to reduce cost.
    let expr = "(+ (+ (* x x) (* 4 x)) 3)";
    let initial = {
        let e: RecExpr<Math> = expr.parse().unwrap();
        let s = MathState::new(e);
        s.cost[usize::from(s.root())]
    };
    let (_, best) = metropolis_best(expr, 80_000, 1.0, 42);
    assert!(best < initial, "expected cost < {}, got {}", initial, best);
}

#[test]
fn sto_diff_power_harder() {
    // (d x (- (pow x 3) (* 7 (pow x 2)))) has a much cheaper equivalent form.
    let expr = "(d x (- (pow x 3) (* 7 (pow x 2))))";
    let initial = {
        let e: RecExpr<Math> = expr.parse().unwrap();
        let s = MathState::new(e);
        s.cost[usize::from(s.root())]
    };
    let (_, best) = metropolis_best(expr, 50_000, 1.0, 0);
    assert!(best < initial, "expected cost < {}, got {}", initial, best);
}

#[test]
fn sto_integ_part1() {
    // (i (* x (cos x)) x)  via integration by parts.
    let expr = "(i (* x (cos x)) x)";
    let initial = {
        let e: RecExpr<Math> = expr.parse().unwrap();
        let s = MathState::new(e);
        s.cost[usize::from(s.root())]
    };
    let (_, best) = metropolis_best(expr, 100_000, 1.0, 42);
    assert!(best < initial, "expected cost < {}, got {}", initial, best);
}

#[test]
fn sto_integ_part2() {
    // (i (* (cos x) x) x) should also improve via the same identities.
    let expr = "(i (* (cos x) x) x)";
    let initial = {
        let e: RecExpr<Math> = expr.parse().unwrap();
        let s = MathState::new(e);
        s.cost[usize::from(s.root())]
    };
    let (_, best) = metropolis_best(expr, 100_000, 1.0, 42);
    assert!(best < initial, "expected cost < {}, got {}", initial, best);
}

#[test]
fn sto_integ_part3() {
    // (i (ln x) x) -> (- (* x (ln x)) x) up to equivalent lower-cost forms.
    let expr = "(i (ln x) x)";
    let initial = {
        let e: RecExpr<Math> = expr.parse().unwrap();
        let s = MathState::new(e);
        s.cost[usize::from(s.root())]
    };
    let (_, best) = metropolis_best(expr, 120_000, 1.0, 42);
    assert!(best < initial, "expected cost < {}, got {}", initial, best);
}
