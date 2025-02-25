use crate::utils::pprint::PrettyPrint;

use std::time;

pub struct DebugEnv {
    debug: bool,
    start: time::Instant
}

impl DebugEnv {
    pub fn print<T: PrettyPrint>(&self, msg: &str, ast: &T) {
        if self.debug {
            let now = time::Instant::now();
            let t = now.duration_since(self.start).as_micros();
            let bounds = "=".repeat(5);
            let ast = ast.pprint_default();
            println!("{0} {msg} (time: {t} us) {0}\n{ast}", bounds);
        }
    }
}

pub fn init(debug_flag: bool) -> DebugEnv {
    DebugEnv {debug: debug_flag, start: time::Instant::now()}
}
