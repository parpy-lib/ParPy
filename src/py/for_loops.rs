use super::from_py;
use super::ast::*;
use crate::py_runtime_error;
use crate::utils::err::CompileError;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

use pyo3::prelude::*;
use pyo3::types::*;
use std::collections::BTreeSet;

fn convert_for_loop_target<'py, 'a>(
    target: Bound<'py, PyAny>,
    i: &Info,
    env: &from_py::ConvertEnv<'py, 'a>
) -> PyResult<Vec<Name>> {
    if target.is_instance(&env.ast.getattr("Name")?)? {
        let s = target.getattr("id")?.extract::<String>()?;
        Ok(vec![Name::new(s)])
    } else if target.is_instance(&env.ast.getattr("Tuple")?)? {
        Ok(target.getattr("elts")?
            .try_iter()?
            .map(|e| convert_for_loop_target(e?, &i, &env))
            .collect::<PyResult<Vec<Vec<Name>>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<Name>>())
    } else {
        py_runtime_error!(i, "The target of a for-loop must be one or more variables.")
    }
}

fn ensure_non_zero_step_size(e: &Expr) -> PyResult<()> {
    match e {
        Expr::Int {v, i, ..} if *v == 0 => {
            py_runtime_error!(i, "For-loop step size must be non-zero")
        },
        _ => Ok(())
    }
}

fn convert_range_arguments<'py, 'a>(
    range_args: Bound<'py, PyAny>,
    i: &Info,
    env: &from_py::ConvertEnv<'py, 'a>
) -> PyResult<(Expr, Expr, Expr)> {
    match range_args.len()? {
        1 => {
            let lo = Expr::Int {v: 0, ty: Type::Unknown, i: i.clone()};
            let hi = from_py::convert_expr(range_args.get_item(0)?, env)?;
            Ok((lo, hi, Expr::Int {v: 1, ty: Type::Unknown, i: i.clone()}))
        },
        2 => {
            let lo = from_py::convert_expr(range_args.get_item(0)?, env)?;
            let hi = from_py::convert_expr(range_args.get_item(1)?, env)?;
            Ok((lo, hi, Expr::Int {v: 1, ty: Type::Unknown, i: i.clone()}))
        },
        3 => {
            let lo = from_py::convert_expr(range_args.get_item(0)?, env)?;
            let hi = from_py::convert_expr(range_args.get_item(1)?, env)?;
            let step = from_py::convert_expr(range_args.get_item(2)?, env)?;
            ensure_non_zero_step_size(&step)?;
            Ok((lo, hi, step))
        }
        _ => py_runtime_error!(i, "Invalid number of arguments passed to range.")
    }
}

fn for_loop_uses_range_builtin<'py, 'a>(
    iter: &Bound<'py, PyAny>,
    env: &from_py::ConvertEnv<'py, 'a>
) -> PyResult<bool> {
    let py = iter.py();
    let func = iter.getattr("func")?;
    if let Ok(e) = from_py::eval_node(&func, env, py) {
        let builtins = py.import("builtins")?;
        if e.eq(builtins.getattr("range")?)? {
            Ok(true)
        } else {
            Ok(false)
        }
    } else {
        Ok(false)
    }
}

fn convert_for_loop_ranges<'py, 'a>(
    iter: &Bound<'py, PyAny>,
    i: &Info,
    env: &from_py::ConvertEnv<'py, 'a>
) -> PyResult<Vec<(Expr, Expr, Expr)>> {
    // Ensure we are using the 'parpy.builtin.ranges' function.
    let py = iter.py();
    let func = iter.getattr("func")?;
    if let Ok(e) = from_py::eval_node(&func, env, py) {
        let parpy = py.import("parpy")?;
        let parpy_builtins = parpy.getattr("builtin")?;
        if e.eq(parpy_builtins.getattr("ranges")?)? {
            Ok(())
        } else {
            py_runtime_error!(i, "???")
        }
    } else {
        py_runtime_error!(i, "???")
    }?;

    iter.getattr("args")?
        .try_iter()?
        .map(|e| {
            let e = e?;
            if e.is_instance(&env.ast.getattr("Tuple")?)? {
                convert_range_arguments(e.getattr("elts")?, i, env)
            } else {
                let e = PyList::new(py, vec![e]).unwrap().into_any();
                convert_range_arguments(e, i, env)
            }
        })
        .collect::<PyResult<Vec<(Expr, Expr, Expr)>>>()
}

fn assert_independent_expr(e: &Expr, vars: &BTreeSet<Name>) -> PyResult<()> {
    match e {
        Expr::Var {id, i, ..} if vars.contains(&id) => {
            py_runtime_error!(
                i,
                "Found for-loop with dependent range bounds, which is not supported."
            )
        },
        _ => e.sfold_result(Ok(()), |_, e| {
            assert_independent_expr(e, vars)
        })
    }
}

fn assert_for_loops_independent_ranges(
    targets: &Vec<Name>,
    ranges: &Vec<(Expr, Expr, Expr)>
) -> PyResult<()> {
    let vars = targets.iter().cloned().collect::<BTreeSet<Name>>();
    ranges.iter()
        .fold(Ok(()), |acc, (lo, hi, step)| {
            if let Ok(_) = acc {
                assert_independent_expr(&lo, &vars)?;
                assert_independent_expr(&hi, &vars)?;
                assert_independent_expr(&step, &vars)
            } else {
                acc
            }
        })
}

fn make_for_loop_iteration_count<'py, 'a>(
    lo: &Expr,
    hi: &Expr,
    step: &Expr,
    i: &Info
) -> Expr {
    let mk_binop = |lhs, op, rhs| Expr::BinOp {
        lhs: Box::new(lhs), op, rhs: Box::new(rhs),
        ty: Type::Unknown, i: i.clone()
    };
    // We compute the upper-bound using the following expression
    //
    //   ceil((hi-lo) / step)
    //
    // which we express as follows using integer division
    //
    //   ((hi - lo) + (step - 1)) / step
    //
    // NOTE(larshum, 2025-12-09): We use this approach under the assumption that all the specified
    // loop bounds are independent. If the bounds of one loop can be dependent on the iteration
    // variable of another loop, it is much more difficult to generate efficient code for it.
    mk_binop(
        mk_binop(
            mk_binop(
                hi.clone(),
                BinOp::Sub,
                lo.clone()
            ),
            BinOp::Add,
            mk_binop(
                step.clone(),
                BinOp::Sub,
                Expr::Int {v: 1, ty: Type::Unknown, i: i.clone()}
            )
        ),
        BinOp::FloorDiv,
        step.clone()
    )
}

fn product_of_exprs(expr_slice: &[Expr], i: &Info) -> Expr {
    if expr_slice.is_empty() {
        Expr::Int {v: 1, ty: Type::Unknown, i: i.clone()}
    } else {
        let fst = expr_slice[0].clone();
        expr_slice[1..].iter()
            .fold(fst, |acc, e| Expr::BinOp {
                lhs: Box::new(acc),
                op: BinOp::Mul,
                rhs: Box::new(e.clone()),
                ty: Type::Unknown,
                i: i.clone()
            })
    }
}

fn generate_for_variable_definition(
    len_exprs: &[Expr],
    k: &Name,
    idx: usize,
    var: Name,
    lo: Expr,
    step: Expr,
    i: &Info
) -> Stmt {
    let k = Expr::BinOp {
        lhs: Box::new(Expr::BinOp {
            lhs: Box::new(Expr::Var {
                id: k.clone(), ty: Type::Unknown, i: i.clone()
            }),
            op: BinOp::FloorDiv,
            rhs: Box::new(product_of_exprs(&len_exprs[idx+1..], &i)),
            ty: Type::Unknown,
            i: i.clone()
        }),
        op: BinOp::Rem,
        rhs: Box::new(len_exprs[idx].clone()),
        ty: Type::Unknown,
        i: i.clone()
    };
    let rhs = Expr::BinOp {
        lhs: Box::new(Expr::BinOp {
            lhs: Box::new(step),
            op: BinOp::Mul,
            rhs: Box::new(k),
            ty: Type::Unknown,
            i: i.clone()
        }),
        op: BinOp::Add,
        rhs: Box::new(lo),
        ty: Type::Unknown,
        i: i.clone()
    };
    Stmt::Definition {
        ty: Type::Unknown,
        id: var,
        expr: rhs,
        labels: vec![],
        i: i.clone()
    }
}

pub fn convert_for_loop<'py, 'a>(
    stmt: Bound<'py, PyAny>,
    i: Info,
    env: &from_py::ConvertEnv<'py, 'a>
) -> PyResult<Stmt> {
    // The targets of the for-loop either consists of a single variable, or a tuple of
    // an arbitrary number of variables.
    let mut targets = convert_for_loop_target(stmt.getattr("target")?, &i, env)?;

    let body = from_py::convert_stmts(stmt.getattr("body")?, env)?;

    // The iterator of the for-loop can either be the builtin 'range' or the special
    // 'parpy.builtin.ranges' which accepts a list of range specifications.
    let iter = stmt.getattr("iter")?;
    if !iter.is_instance(&env.ast.getattr("Call")?)? {
        py_runtime_error!(i, "For-loop must iterate using 'range' or 'parpy.builtin.ranges'.")?
    };
    let (var, lo, hi, step, body) = if for_loop_uses_range_builtin(&iter, env)? {
        if targets.len() == 1 {
            let var = targets.pop().unwrap();
            let (lo, hi, step) = convert_range_arguments(iter.getattr("args")?, &i, env)?;
            Ok((var, lo, hi, step, body))
        } else {
            py_runtime_error!(i, "The range builtin expects one target variable.\n\
                                  Use the 'parpy.builtin.ranges' primitive to \
                                  specify a loop across multiple variables.")
        }
    } else {
        let ranges = convert_for_loop_ranges(&iter, &i, env)?;
        if targets.len() == ranges.len() {
            let k = Name::sym_str("combined_var");
            // Ensure that the bounds of all for-loops are independent from each other, in the
            // sense that they do not depend on any target variables.
            assert_for_loops_independent_ranges(&targets, &ranges)?;
            let len_exprs = ranges.iter()
                .map(|(lo, hi, step)| {
                    make_for_loop_iteration_count(lo, hi, step, &i)
                })
                .collect::<Vec<Expr>>();
            let var_defs = targets.into_iter()
                .enumerate()
                .zip(ranges.into_iter())
                .map(|((idx, var), (lo, _, step))| {
                    generate_for_variable_definition(
                        &len_exprs, &k, idx, var, lo, step, &i
                    )
                })
                .collect::<Vec<Stmt>>();
            let lo = Expr::Int {v: 0, ty: Type::Unknown, i: i.clone()};
            let hi = product_of_exprs(&len_exprs[..], &i);
            let step = Expr::Int {v: 1, ty: Type::Unknown, i: i.clone()};
            let body = var_defs.into_iter()
                .chain(body.into_iter())
                .collect::<Vec<Stmt>>();
            Ok((k, lo, hi, step, body))
        } else {
            py_runtime_error!(i, "For-loop was declared with {0} target variables, \
                                  but was provided {1} ranges.",
                                  targets.len(), ranges.len())
        }
    }?;

    if stmt.getattr("orelse")?.len()? == 0 {
        Ok(Stmt::For {var, lo, hi, step, body, labels: vec![], i})
    } else {
        py_runtime_error!(i, "For-loops with an else-clause are not supported.")
    }
}
