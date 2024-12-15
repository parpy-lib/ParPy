use std::fs;

// The info field contains information referring back to the source file an AST node was parsed
// from.
#[derive(Clone, Debug)]
pub struct Info {
    filename : String,
    l1 : usize,
    l2 : usize,
    c1 : usize,
    c2 : usize
}

impl Info {
    pub fn new(fname : &str, l1 : usize, l2 : usize, c1 : usize, c2 : usize) -> Info {
        let filename = fname.to_string();
        Info {filename, l1, l2, c1, c2}
    }

    pub fn error_msg(&self, msg : String) -> String {
        if let Ok(code) = fs::read_to_string(&self.filename) {
            self.extract_lines(code, msg)
        } else {
            msg
        }
    }

    fn extract_lines(&self, code : String, msg : String) -> String {
        let select_lines = code.lines()
            .skip(self.l1 - 1)
            .take(self.l2 - self.l1 + 1)
            .collect::<Vec<&str>>()
            .join("\n");
        let err_markers = format!("{0}{1}\n", " ".repeat(self.c1), "^".repeat(self.c2 - self.c1));
        format!(
            "{msg}\n\nIn span [{0},{1}:{2},{3}] of file {4}:\n{select_lines}\n{err_markers}",
            self.l1, self.c1, self.l2, self.c2, self.filename
        )
    }
}

impl Default for Info {
    fn default() -> Info {
        Info {filename : "<generated>".to_string(), l1 : 0, l2 : 0, c1 : 0, c2 : 0}
    }
}

pub trait InfoNode {
    fn get_info(&self) -> Info;
}
