//! Reading progress: a hairline along the top and a small pill in the corner that says how far
//! you are and how long is left. Hover the pill for words read, words left and your pace.
//!
//! Scrolling never re-runs the view: the scroll handler updates live values (the bar's rect, the
//! pill's text) and the compositor reads the fraction for the pill's dots.

use crate::{
    State,
    document::{grouped, minutes, short_minutes},
    reading::Reading,
    ui::{
        fx,
        theme::{DONE, EDGE, INK, MUTED, PANEL, c, figures, kit},
        widgets::label,
    },
};
use rune::{El, Live, LiveText, ScrollPos, col, effect, el, row, text};
use std::{cell::RefCell, rc::Rc, sync::OnceLock, time::Instant};

const BAR: f32 = 2.0;

/// The live values the scroll handler writes.
#[derive(Clone)]
pub struct Hud {
    bar: Live<[f32; 4]>,
    percent: LiveText,
    left: LiveText,
    read: LiveText,
    pace: LiveText,
}

impl Default for Hud {
    fn default() -> Self {
        Self {
            bar: Live::new([0.0, 0.0, 0.0, BAR]),
            percent: LiveText::new("0%"),
            left: LiveText::new(""),
            read: LiveText::new(""),
            pace: LiveText::new(""),
        }
    }
}

fn seconds() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

impl Hud {
    fn show(&self, r: &Reading, width: f32, prior: f32) {
        self.bar.set([0.0, 0.0, (r.fraction * width).max(0.0), BAR]);
        self.percent
            .set(format!("{:.0}%", (r.fraction * 100.0).floor()));
        self.left.set(if r.words_left() == 0 {
            "the end".to_owned()
        } else {
            format!("{} left", short_minutes(r.seconds_left()))
        });
        self.read.set(format!(
            "{} of {} words",
            grouped(r.words_read()),
            grouped(r.words)
        ));
        let wpm = r.wpm().round();
        self.pace.set(if r.confident() {
            format!("{wpm} words a minute, your pace")
        } else if (wpm - f64::from(prior)).abs() < 1.0 && prior != crate::settings::DEFAULT_WPM {
            format!("{wpm} words a minute, from last time")
        } else {
            format!("{wpm} words a minute, measuring…")
        });
    }

    /// The scroll handler: follows the article's position into the progress and live values.
    pub fn follow(
        &self,
        reading: &Rc<RefCell<Option<Reading>>>,
        prior: f32,
    ) -> impl Fn(ScrollPos) + 'static {
        let (hud, reading) = (self.clone(), reading.clone());
        move |at: ScrollPos| {
            if let Some(r) = reading.borrow_mut().as_mut() {
                r.observe(seconds(), at.offset[1], at.max[1]);
                hud.show(r, at.viewport[0], prior);
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
    let hud = &s.reader.hud;
    let finished = s.reader.finished_at;
    let dots = effect(fx::STRIP)
        .key("reading-dots")
        .effect_data([finished.map_or(-1.0, |at| at as f32), 0.0, 0.0, 0.0])
        .size(60.0, 12.0);
    let small = figures(12.5, 500.0);
    let summary = if finished.is_some() {
        row()
            .key("pill-finished")
            .child(label("Finished", small, INK))
    } else {
        row()
            .key("pill-reading")
            .gap(8.0)
            .center_y()
            .child(
                text("100%")
                    .live_text(&hud.percent)
                    .font(small)
                    .color(c(INK))
                    .align(rune::Align::End),
            )
            .child(label("·", small, MUTED))
            .child(
                text("59 min left")
                    .live_text(&hud.left)
                    .font(small)
                    .color(c(MUTED)),
            )
    };
    let details = col()
        .key("pill-details")
        .pin(None, None, Some(0.0), Some(44.0))
        .w(250.0)
        .px(14.0)
        .py(10.0)
        .gap(3.0)
        .radius(10.0)
        .fill(c(PANEL))
        .border(1.0, c(EDGE))
        .group_opacity(0.0, 1.0)
        .child(
            text("123,456 of 123,456 words")
                .live_text(&hud.read)
                .font(kit().ty.ui_medium)
                .color(c(INK)),
        )
        .child(
            text("999 words a minute, from last time")
                .live_text(&hud.pace)
                .font(kit().ty.text_xsmall)
                .color(c(MUTED)),
        );
    row()
        .key("reading-pill")
        .role("status")
        .label("Reading progress")
        .hover_group()
        .h(34.0)
        .px(12.0)
        .gap(10.0)
        .center_y()
        .radius(17.0)
        .fill(c(PANEL))
        .border(1.0, c(EDGE))
        .child(details)
        .child(dots)
        .child(summary)
        .pin(None, None, Some(22.0), Some(20.0))
}

/// The closing line under the transcript, once you've read to the end.
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
