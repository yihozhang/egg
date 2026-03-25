//! Stochastic search tests using a subset of the Math language.
//!
//! Each test starts from an expression and checks that greedy MH descent
//! (temperature = 0) finds a lower-cost expression within a fixed step budget.

use std::sync::Arc;

use egg::{
    stochastic::{
        ConstantTemp, MhRunner, SimpleLcg, State, StoAnalysis, StoConditionalApplier,
        StoRewrite,
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
            Math::Diff(..) | Math::Integral(..) => 100.0,
            _ => 1.0,
        };
        op + enode.fold(0.0, |acc, c| acc + children_cost[usize::from(c)])
    }
}

type MathState = State<Math, MathCost>;
type MathRw = StoRewrite<Math, MathCost>;

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
        StoConditionalApplier { applier: Arc::new(p(rhs)), condition: Box::new(cond) },
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
    move |s: &MathState, _: Id, subst: &Subst| {
        matches!(s.rec_expr[subst[v]], Math::Constant(_))
    }
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

// ─── Driver ──────────────────────────────────────────────────────────────────

/// Run pure greedy descent (temperature = 0) from `start` for up to `n_steps`
/// steps.  Returns the best expression and its cost found during the run.
fn greedy_best(start: &str, n_steps: u64) -> (RecExpr<Math>, f64) {
    let expr: RecExpr<Math> = start.parse().unwrap();
    let state = MathState::new(expr);
    let mut runner = MhRunner::new(state, rules());
    let mut rng = SimpleLcg::new(42);
    runner.run(n_steps, &ConstantTemp(0.0), &mut rng);
    (runner.best_expr.clone(), runner.best_cost)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn sto_diff_same() {
    // (d x x) → 1  via d-variable  (cost 102 → 1)
    let (_, cost) = greedy_best("(d x x)", 500);
    assert_eq!(cost, 1.0);
}

#[test]
fn sto_diff_different() {
    // (d x y) → 0  via d-constant  (cost 102 → 1)
    let (_, cost) = greedy_best("(d x y)", 500);
    assert_eq!(cost, 1.0);
}

#[test]
fn sto_zero_add() {
    // (+ x 0) → x  via zero-add  (cost 3 → 1)
    let (_, cost) = greedy_best("(+ x 0)", 500);
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
    let (_, best) = greedy_best("(* (pow 2 x) (pow 2 y))", 2000);
    assert!(best < initial, "cost did not decrease: {} vs initial {}", best, initial);
}

#[test]
fn sto_diff_sin() {
    // (d x (sin x)) → (cos x)  via d-sin  (cost 103 → 2)
    let (_, cost) = greedy_best("(d x (sin x))", 500);
    assert_eq!(cost, 2.0);
}

#[test]
fn sto_integ_cos() {
    // (i (cos x) x) → (sin x)  via i-cos  (cost 103 → 2)
    let (_, cost) = greedy_best("(i (cos x) x)", 500);
    assert_eq!(cost, 2.0);
}
