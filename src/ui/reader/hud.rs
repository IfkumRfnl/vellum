//! Reading progress: how far you are and how long is left at your pace, at the left of the
//! toolbar, over a hairline that fills across the top of the page.
//!
//! Scrolling never re-runs the view: the scroll handler writes live values (the line's rect and
//! the status text).

use crate::{
    State,
    document::short_minutes,
    reading::Reading,
    ui::theme::{EDGE, INK, MUTED, c, figures},
};
use rune::{El, Live, LiveText, ScrollPos, el, text};
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
        let percent = (r.fraction * 100.0).floor();
        self.line.set(if r.words_left() == 0 {
            format!("{percent:.0}%")
        } else {
            format!("{percent:.0}%  ·  {} left", short_minutes(r.seconds_left()))
        });
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

/// How far you are and how long is left, for the toolbar.
pub fn status(s: &State) -> El {
    text("100%  ·  59 min left")
        .key("reading-status")
        .role("status")
        .label("Reading progress")
        .live_text(&s.reader.hud.line)
        .font(figures(13.0, 500.0))
        .color(c(MUTED))
}

/// The hairline across the top of the page.
pub fn bar(s: &State) -> El {
    let fill = el()
        .key("reading-bar-fill")
        .h(BAR)
        .w_full()
        .absolute(0.0, 0.0)
        .fill(c(INK))
        .live_rect(&s.reader.hud.bar);
    el().key("reading-bar")
        .w_full()
        .h(BAR)
        .fill(c(EDGE).alpha(0.5))
        .child(fill)
}
