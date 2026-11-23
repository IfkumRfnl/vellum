//! Reading progress: a hairline along the top, and a small pill in the corner with how far you
//! are, how long is left and your reading pace.
//!
//! Scrolling never re-runs the view: the scroll handler writes live values (the bar's rect and
//! the pill's line).

use crate::{
    State,
    document::{grouped, minutes, short_minutes},
    reading::Reading,
    ui::{
        theme::{DONE, EDGE, INK, MUTED, PANEL, c, figures, kit},
        widgets::label,
    },
};
use rune::{El, Live, LiveText, ScrollPos, el, row, text};
use std::{cell::RefCell, rc::Rc, sync::OnceLock, time::Instant};

const BAR: f32 = 2.0;

/// The live values the scroll handler writes.
#[derive(Clone)]
pub struct Hud {
    bar: Live<[f32; 4]>,
    line: LiveText,
}

impl Default for Hud {
    fn default() -> Self {
        Self {
            bar: Live::new([0.0, 0.0, 0.0, BAR]),
            line: LiveText::new(""),
        }
    }
}

fn seconds() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

impl Hud {
    fn show(&self, r: &Reading, width: f32) {
        self.bar.set([0.0, 0.0, (r.fraction * width).max(0.0), BAR]);
        let left = if r.words_left() == 0 {
            "at the end".to_owned()
        } else {
            format!("{} left", short_minutes(r.seconds_left()))
        };
        self.line.set(format!(
            "{:.0}%  ·  {left}  ·  {:.0} wpm",
            (r.fraction * 100.0).floor(),
            r.wpm()
        ));
    }

    /// The scroll handler: follows the article's position into the progress and live values.
    pub fn follow(&self, reading: &Rc<RefCell<Option<Reading>>>) -> impl Fn(ScrollPos) + 'static {
        let (hud, reading) = (self.clone(), reading.clone());
        move |at: ScrollPos| {
            if let Some(r) = reading.borrow_mut().as_mut() {
                r.observe(seconds(), at.offset[1], at.max[1]);
                hud.show(r, at.viewport[0]);
            }
        }
    }
}

/// The hairline across the top of the reader.
pub fn bar(s: &State) -> El {
    let fill = el()
        .key("reading-bar-fill")
        .h(BAR)
        .w_full()
        .absolute(0.0, 0.0)
        .fill(c(if s.reader.finished_at.is_some() {
            DONE
        } else {
            INK
        }))
        .live_rect(&s.reader.hud.bar);
    el().key("reading-bar")
        .w_full()
        .h(BAR)
        .fill(c(EDGE).alpha(0.5))
        .child(fill)
}

/// The pill in the corner.
pub fn pill(s: &State) -> El {
    let style = figures(12.5, 500.0);
    let content = if s.reader.finished_at.is_some() {
        label("Finished", style, DONE)
    } else {
        text("100%  ·  59 min left  ·  888 wpm")
            .live_text(&s.reader.hud.line)
            .font(style)
            .color(c(MUTED))
    };
    row()
        .key("reading-pill")
        .role("status")
        .label("Reading progress")
        .h(32.0)
        .px(14.0)
        .center_y()
        .radius(16.0)
        .fill(c(PANEL))
        .border(1.0, c(EDGE))
        .child(content)
        .pin(None, None, Some(22.0), Some(20.0))
}

pub fn finish_note(s: &State) -> Option<El> {
    s.reader.finished_at?;
    let reading = s.reader.progress.borrow();
    let r = reading.as_ref()?;
    Some(
        label(
            format!(
                "You reached the end · {} words · about {} at {:.0} words a minute",
                grouped(r.words),
                minutes(r.words as f64 / r.wpm() * 60.0),
                r.wpm()
            ),
            kit().ty.text_xsmall,
            MUTED,
        )
        .wrap(),
    )
}
