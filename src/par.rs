use crate::utils::pprint::{PrettyPrint, PrettyPrintEnv};

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

pub const REDUCE_PAR_LABEL: &'static str = "_reduce";
pub const DEFAULT_TPB: i64 = 1024;

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct LoopPar {
    pub nthreads: i64,
    pub reduction: bool,
    pub tpb: i64
}

impl Default for LoopPar {
    fn default() -> Self {
        LoopPar {nthreads: 0, reduction: false, tpb: DEFAULT_TPB}
    }
}

impl PrettyPrint for LoopPar {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let LoopPar {nthreads, reduction, tpb} = self;
        (env, format!("{{nthreads = {nthreads}, reduction = {reduction}, tpb = {tpb}}}"))
    }
}

#[pymethods]
impl LoopPar {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        format!("{self:?}")
    }

    #[new]
    fn _loop_par_new() -> Self {
        LoopPar::default()
    }

    pub fn is_parallel(&self) -> bool {
        self.nthreads > 0
    }

    pub fn threads(&self, nthreads: i64) -> PyResult<Self> {
        if self.nthreads == 0 {
            Ok(LoopPar {nthreads, ..self.clone()})
        } else {
            Err(PyRuntimeError::new_err("The number of threads can only be set once"))
        }
    }

    pub fn reduce(&self) -> Self {
        LoopPar {reduction: true, ..self.clone()}
    }

    pub fn tpb(&self, tpb: i64) -> PyResult<Self> {
        if tpb > 0 && tpb % 32 == 0 {
            Ok(LoopPar {tpb, ..self.clone()})
        } else {
            Err(PyRuntimeError::new_err("The number of threads per block must \
                                         be a positive integer divisible by 32."))
        }
    }
}

fn merge_values(l: i64, r: i64, default_value: i64) -> Option<i64> {
    if l == default_value || l == r {
        Some(r)
    } else if r == default_value {
        Some(l)
    } else {
        None
    }
}

impl LoopPar {

    pub fn try_merge(mut self, other: Option<&LoopPar>) -> Option<LoopPar> {
        if let Some(o) = other {
            self.nthreads = merge_values(self.nthreads, o.nthreads, 0)?;
            self.tpb = merge_values(self.tpb, o.tpb, DEFAULT_TPB)?;
            self.reduction = self.reduction || o.reduction;
        };
        Some(self)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;

    #[test]
    fn set_threads_once() {
        let th = LoopPar::default().threads(128);
        assert!(th.is_ok());
        assert_eq!(th.unwrap().nthreads, 128);
    }

    #[test]
    fn set_threads_twice() {
        let r = LoopPar::default().threads(128).unwrap().threads(256);
        assert_py_error_matches(r, "number of threads");
    }

    #[test]
    fn set_tpb_non_multiple_of_warp_size() {
        let r = LoopPar::default().tpb(16);
        assert_py_error_matches(r, "number of threads per block.*divisible by 32");
    }

    #[test]
    fn merge_equal_values() {
        assert_eq!(merge_values(10, 10, 5), Some(10));
    }

    #[test]
    fn merge_with_default() {
        assert_eq!(merge_values(10, 5, 10), Some(5));
        assert_eq!(merge_values(5, 10, 10), Some(5));
    }

    #[test]
    fn merge_inequal_values() {
        assert_eq!(merge_values(1, 2, 3), None);
    }

    #[test]
    fn par_is_seq() {
        assert!(!LoopPar::default().is_parallel());
    }

    #[test]
    fn par_is_parallel() {
        assert!(LoopPar::default().threads(2).unwrap().is_parallel());
    }

    fn loop_par1() -> LoopPar {
        LoopPar::default().threads(64).unwrap()
    }

    fn loop_par2() -> LoopPar {
        LoopPar::default().threads(128).unwrap()
    }

    fn loop_par3() -> LoopPar {
        LoopPar::default().threads(128).unwrap().tpb(64).unwrap()
    }

    #[test]
    fn merge_none() {
        assert_eq!(loop_par1().try_merge(None), Some(loop_par1()));
    }

    #[test]
    fn merge_equal_pars() {
        assert_eq!(loop_par1().try_merge(Some(&loop_par1())), Some(loop_par1()));
    }

    #[test]
    fn merge_distinct_threads_par() {
        assert_eq!(loop_par1().try_merge(Some(&loop_par2())), None);
    }

    #[test]
    fn merge_distinct_tpb_par() {
        assert_eq!(loop_par2().try_merge(Some(&loop_par3())), Some(loop_par3()));
    }

    #[test]
    fn merge_equal_reduction() {
        let p1 = loop_par2();
        let p2 = loop_par2().reduce();
        assert_eq!(p1.try_merge(Some(&p2.clone())), Some(p2));
    }
}
