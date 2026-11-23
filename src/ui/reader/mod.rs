//! The transcript, as one continuous page.
//!
//! Paragraphs are virtual rows, so a three-hour podcast scrolls as lightly as a short clip:
//! only what's near the viewport is laid out. The title is the first row.

mod appearance;
mod hud;

pub use hud::Hud;

use crate::{
    Document, State,
    document::{grouped, minutes, timestamp},
    ui::{
        theme::{INK, MUTED, SELECTION, c, kit},
        widgets::{Tone, button, label, line},
    },
};
use rune::{
    El, Family, GrowthSpec, Key, KeyEvent, ScrollSpec, TextStyle, Tx, WheelEvent, col, row, text,
};
use rune_kit::{ScrollBar, ScrollBarStyle};
use std::borrow::Cow;

const SERIF: Family = Family::Named("Georgia, Constantia, DejaVu Serif");
const FACES: [Family; 3] = [SERIF, Family::Sans, Family::SystemMono];
/// The article's scroll viewport.
const ARTICLE: &str = "article";

pub fn toolbar(s: &State) -> El {
    let copied = s.reader.copied_at.is_some();
    let actions = row()
        .gap(8.0)
        .child(
            button(
                if copied { "Copied" } else { "Copy" },
                Tone::Quiet,
                |s, tx| {
                    if let Some(doc) = &s.doc {
                        tx.copy(doc.plain_text());
                        s.reader.copied_at = Some(s.clock);
                    }
                },
            )
            .key("copy")
            .label("Copy transcript")
            .help("Copy the whole transcript"),
        )
        .child(
            button(
                "Aa",
                if s.reader.appearance {
                    Tone::Primary
                } else {
                    Tone::Quiet
                },
                |s, tx| {
                    tx.instant();
                    s.reader.appearance = !s.reader.appearance;
                },
            )
            .label("Reading appearance")
            .help("Reading appearance"),
        )
        .child(button("New video", Tone::Quiet, |s, tx| s.new_video(tx)));
    row()
        .w_full()
        .justify_between()
        .center_y()
        .child(hud::status(s))
        .child(actions)
}

pub fn view(s: &State, doc: &Document) -> El {
    let width = s.settings.reading_width.points();
    let count = doc.paragraphs.len() + 1;
    let list = col()
        .key("paragraphs")
        .w_full()
        .growth(GrowthSpec::default())
        .selection_group(true)
        .virtual_rows(
            count,
            s.settings.size as f32 * 6.5,
            |_: &State, i| -> Cow<'_, str> {
                match i {
                    0 => "head".into(),
                    i => format!("p-{}", i - 1).into(),
                }
            },
            |s: &State, i| {
                let doc = s.doc.as_ref().expect("reader has a transcript");
                match i {
                    0 => head(s, doc),
                    i => paragraph(s, doc, i - 1),
                }
            },
        );
    let article = col()
        .key(ARTICLE)
        .grow(1.0)
        .min_h(0.0)
        .scroll(ScrollSpec {
            stick_bottom: false,
        })
        .on_scroll_offset(s.reader.hud.follow(&s.reader.progress))
        .on_wheel(font_wheel)
        .child(
            col()
                .w_full()
                .items_center()
                .px(26.0)
                .pt(34.0)
                .pb(64.0)
                .child(
                    col()
                        .key("article-column")
                        .w_full()
                        .max_w(width)
                        .child(list),
                ),
        )
        .scroll_bar_styled(ScrollBarStyle::native(c("#3a3a3a"), c("#6b6b6b")).inside());

    let mut pane = col().key("reader").grow(1.0).min_h(0.0).child(hud::bar(s));
    if s.reader.appearance {
        pane = pane.child(
            col()
                .key("appearance-row")
                .w_full()
                .items_center()
                .px(26.0)
                .py(10.0)
                .child(appearance::panel(s).max_w(660.0)),
        );
    }
    pane.child(article)
}

fn style(s: &State) -> TextStyle {
    let size = s.settings.size as f32;
    TextStyle::new(size, 400.0)
        .family(FACES[s.settings.font])
        .lh(size * if s.settings.spacious { 1.65 } else { 1.36 })
}

fn head(s: &State, doc: &Document) -> El {
    let ty = &kit().ty;
    let words = doc.words();
    let length = format!(
        "{} words, {} read",
        grouped(words),
        minutes(words as f64 / f64::from(s.settings.reading_wpm) * 60.0)
    );
    let meta = [&doc.channel, &doc.source, &doc.language, &doc.speaker_note]
        .into_iter()
        .filter(|v| !v.is_empty())
        .map(String::as_str)
        .chain([length.as_str()])
        .collect::<Vec<_>>()
        .join("   ·   ");
    col()
        .w_full()
        .gap(20.0)
        .pb(24.0)
        .child(
            label(
                &doc.title,
                TextStyle::new(36.0, 500.0)
                    .family(FACES[s.settings.font])
                    .tracking(-0.65),
                INK,
            )
            .wrap()
            .selectable(true)
            .selection_color(c(SELECTION)),
        )
        .child(label(meta, ty.ui, MUTED).wrap())
        .child(line())
}

fn paragraph(s: &State, doc: &Document, index: usize) -> El {
    let p = &doc.paragraphs[index];
    let small = kit().ty.text_xsmall;
    let mut block = col().w_full().gap(5.0).pb(24.0);
    let mut note = row().gap(0.0).center_y();
    let mut has_note = false;
    if s.settings.timestamps {
        has_note = true;
        let mut stamp = text(timestamp(p.start)).font(small).color(c(MUTED));
        if let Some(link) = doc.link_at(p.start) {
            stamp = stamp
                .key(format!("t-{index}"))
                .role("link")
                .help("Watch from here on YouTube")
                .cursor(rune::Cursor::Pointer)
                .on_press_tx(move |_: &mut State, tx| {
                    let _ = tx.open_url(&link);
                });
        }
        note = note.child(stamp);
    }
    if let Some(name) = &p.speaker {
        if has_note {
            note = note.child(label("   ·   ", small, MUTED));
        }
        has_note = true;
        note = note.child(label(name, small, MUTED));
    }
    if has_note {
        block = block.child(note);
    }
    let body = text(&p.text)
        .font(style(s))
        .color(c(INK))
        .wrap()
        .selection_color(c(SELECTION))
        .selectable(true);
    block.child(body)
}

pub fn font_wheel(s: &mut State, e: WheelEvent, _: &mut Tx) -> bool {
    if !e.mods.ctrl {
        s.reader.wheel_remainder = 0.0;
        return false;
    }
    if !e.momentum && e.dy != 0.0 {
        let steps = if e.precise && !e.notch {
            s.reader.wheel_remainder += e.dy / 40.0;
            let whole = s.reader.wheel_remainder.trunc();
            s.reader.wheel_remainder -= whole;
            whole as i32
        } else {
            s.reader.wheel_remainder = 0.0;
            e.dy.signum() as i32
        };
        s.adjust_font(steps);
    }
    true
}

/// Space pages down (Shift+Space up); Home and End glide to either end.
pub fn on_key(_: &mut State, e: &KeyEvent, tx: &mut Tx) -> bool {
    match &e.key {
        Key::Text(t) if t == " " => tx.scroll_page(ARTICLE, if e.mods.shift { -1 } else { 1 }),
        Key::Home => tx.scroll_to_with(ARTICLE, 0.0, 0.0, rune::Spring::new(0.6, 0.0)),
        Key::End => tx.scroll_to_with(ARTICLE, 0.0, f32::MAX, rune::Spring::new(0.6, 0.0)),
        _ => return false,
    }
    true
}
