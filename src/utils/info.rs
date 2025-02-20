use itertools::Itertools;

use std::fs;

#[derive(Clone, Debug, PartialEq)]
pub struct FilePos {
    pub line: usize,
    pub col: usize
}

impl FilePos {
    pub fn new(line: usize, col: usize) -> FilePos {
        FilePos {line, col}
    }

    pub fn with_line_offset(self, ofs: usize) -> FilePos {
        FilePos {line: self.line + ofs, col: self.col}
    }

    pub fn with_column_offset(self, ofs: usize) -> FilePos {
        FilePos {line: self.line, col: self.col + ofs}
    }

    pub fn merge(l: FilePos, r: FilePos) -> FilePos {
        FilePos {
            line: l.line.min(r.line),
            col: l.col.min(r.col)
        }
    }
}

impl Default for FilePos {
    fn default() -> FilePos {
        FilePos {line: 0, col: 0}
    }
}

// The info field contains information referring back to the source file an AST node was parsed
// from.
#[derive(Clone, Debug, PartialEq)]
pub struct Info {
    filename: String,
    start: FilePos,
    end: FilePos
}

impl Info {
    pub fn new(fname: &str, start: FilePos, end: FilePos) -> Info {
        let filename = fname.to_string();
        Info {filename, start, end}
    }

    pub fn with_file(self, fname: &str) -> Info {
        Info {filename: fname.to_string(), ..self}
    }

    pub fn with_line_offset(self, ofs: usize) -> Info {
        let start = self.start.with_line_offset(ofs);
        let end = self.end.with_line_offset(ofs);
        Info {start, end, ..self}
    }

    pub fn with_column_offset(self, ofs: usize) -> Info {
        let start = self.start.with_column_offset(ofs);
        let end = self.end.with_column_offset(ofs);
        Info {start, end, ..self}
    }

    pub fn merge(l: Info, r: Info) -> Info {
        let filename = if l.filename == r.filename {
            l.filename.clone()
        } else {
            "<unknown>".to_string()
        };
        Info {
            filename,
            start: FilePos::merge(l.start, r.start),
            end: FilePos::merge(l.end, r.end),
        }
    }

    pub fn error_msg(&self, msg: String) -> String {
        if let Ok(code) = fs::read_to_string(&self.filename) {
            self.extract_lines(code, msg)
        } else {
            msg
        }
    }

    fn extract_lines(&self, code: String, msg: String) -> String {
        let start = &self.start;
        let end = &self.end;
        let select_lines = code.lines()
            .skip(start.line - 1)
            .take(end.line - start.line + 1)
            .join("\n");
        let max_col = start.col.max(end.col);
        let min_col = start.col.min(end.col);
        let err_markers = format!(
            "{0}{1}\n",
            " ".repeat(start.col),
            "^".repeat(max_col - min_col)
        );
        let lines_msg = if start.line == end.line {
            format!("line {0}", start.line)
        } else {
            format!("lines {0}-{1}", start.line, end.line)
        };
        format!(
            "{msg}\n\nOn {lines_msg} of file {0}:\n{select_lines}\n{err_markers}",
            self.filename
        )
    }
}

impl Default for Info {
    fn default() -> Info {
        let start = FilePos::default();
        let end = FilePos::default();
        Info {filename: String::new(), start, end}
    }
}

pub trait InfoNode {
    fn get_info(&self) -> Info;
}
