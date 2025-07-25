use crate::option::CompileOptions;
use crate::utils::pprint::PrettyPrint;

use std::time;

pub struct DebugEnv {
    debug_print: bool,
    start: time::Instant
}

impl DebugEnv {
    pub fn new(opts: &CompileOptions) -> DebugEnv {
        DebugEnv {
            debug_print: opts.debug_print,
            start: time::Instant::now()
        }
    }

    fn print_ast_message<T: PrettyPrint>(
        start: time::Instant,
        bounds: &str,
        msg: &str,
        ast: &T
    ) -> String {
        let now = time::Instant::now();
        let t = now.duration_since(start).as_micros();
        format!("{0} {msg} (time: {1} us) {0}\n{2}", bounds, t, ast.pprint_default())
    }

    pub fn print<T: PrettyPrint>(&self, msg: &str, ast: &T) {
        let bounds = "=".repeat(5);
        if self.debug_print {
            let s = DebugEnv::print_ast_message(self.start, &bounds, msg, ast);
            println!("{}", s);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::cuda::ast;

    use regex::Regex;

    #[test]
    fn test_print_message() {
        let start = time::Instant::now();
        let msg = "x";
        let ast: Vec<ast::Top> = vec![];
        let s = DebugEnv::print_ast_message(start, "=", msg, &ast);
        let re = Regex::new(r"= x \(time: \d+ us\) =\n").unwrap();
        assert!(re.is_match(&s));
    }
}
