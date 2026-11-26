//! Markdown tables for reports.

use std::fmt;

/// A Markdown table: columns whose cells all read as numbers ("12.5", "-3%", "1.8×", "42 ms")
/// are right-aligned, the rest left-aligned, and every column is padded so the raw text lines
/// up too.
#[derive(Debug, Clone, Default)]
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    pub fn new<S: ToString>(headers: impl IntoIterator<Item = S>) -> Table {
        Table {
            headers: headers.into_iter().map(|h| h.to_string()).collect(),
            rows: Vec::new(),
        }
    }

    /// Adds a row; missing cells are empty and extra ones dropped.
    pub fn row<S: ToString>(&mut self, cells: impl IntoIterator<Item = S>) -> &mut Table {
        let mut row: Vec<String> = cells.into_iter().map(|c| c.to_string()).collect();
        row.resize(self.headers.len(), String::new());
        self.rows.push(row);
        self
    }

    /// The table as Markdown, ending in a newline.
    pub fn markdown(&self) -> String {
        self.to_string()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    fn numeric(&self, column: usize) -> bool {
        let mut any = false;
        for row in &self.rows {
            let cell = row[column].trim();
            if cell.is_empty() || matches!(cell, "—" | "–" | "-") {
                continue;
            }
            if !looks_numeric(cell) {
                return false;
            }
            any = true;
        }
        any
    }
}

/// A number, perhaps signed, with thousands commas, a leading "~", "≤", "±", "×" or "$", and a
/// unit after it.
fn looks_numeric(cell: &str) -> bool {
    let s = cell.trim_start_matches(['~', '≤', '≥', '<', '>', '±', '+', '-', '−', '$', '×']);
    let digits: String = s
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | ','))
        .filter(|c| *c != ',')
        .collect();
    if digits.parse::<f64>().is_err() {
        return false;
    }
    let unit = s[s
        .char_indices()
        .find(|(_, c)| !(c.is_ascii_digit() || matches!(c, '.' | ',')))
        .map_or(s.len(), |(i, _)| i)..]
        .trim();
    unit.chars().count() <= 4 && !unit.contains(' ') || unit.starts_with('(') && unit.ends_with(')')
}

fn escape(cell: &str) -> String {
    cell.replace('|', "\\|").replace('\n', " ")
}

impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let columns = self.headers.len();
        let headers: Vec<String> = self.headers.iter().map(|h| escape(h)).collect();
        let rows: Vec<Vec<String>> = self
            .rows
            .iter()
            .map(|r| r.iter().map(|c| escape(c)).collect())
            .collect();
        let right: Vec<bool> = (0..columns).map(|c| self.numeric(c)).collect();
        let width: Vec<usize> = (0..columns)
            .map(|c| {
                rows.iter()
                    .map(|r| r[c].chars().count())
                    .chain([headers[c].chars().count(), 3])
                    .max()
                    .unwrap()
            })
            .collect();
        let line = |f: &mut fmt::Formatter<'_>, cells: &[String]| -> fmt::Result {
            write!(f, "|")?;
            for (c, cell) in cells.iter().enumerate() {
                let pad = width[c] - cell.chars().count();
                if right[c] {
                    write!(f, " {}{cell} |", " ".repeat(pad))?;
                } else {
                    write!(f, " {cell}{} |", " ".repeat(pad))?;
                }
            }
            writeln!(f)
        };
        line(f, &headers)?;
        write!(f, "|")?;
        for c in 0..columns {
            if right[c] {
                write!(f, " {}: |", "-".repeat(width[c] - 1))?;
            } else {
                write!(f, " {} |", "-".repeat(width[c]))?;
            }
        }
        writeln!(f)?;
        for row in &rows {
            line(f, row)?;
        }
        Ok(())
    }
}
