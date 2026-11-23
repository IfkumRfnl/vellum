//! The transcript, as one continuous page.
//!
//! Paragraphs are virtual rows, so a three-hour podcast scrolls as lightly as a short clip:
//! only what's near the viewport is laid out. The title is the first row and the closing
//! actions the last.

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
    text_input,
};
use std::borrow::Cow;

const SERIF: Family = Family::Named("Georgia, Constantia, DejaVu Serif");
const FACES: [Family; 3] = [SERIF, Family::Sans, Family::SystemMono];
/// The article's scroll viewport.
const ARTICLE: &str = "article";

pub fn toolbar(s: &State) -> El {
    let editing = s.reader.edits.is_some();
    row()
        .gap(8.0)
        .child(button(
            if editing { "Done" } else { "Edit" },
            if editing { Tone::Primary } else { Tone::Quiet },
            |s, tx| {
                tx.instant();
                s.toggle_editing();
            },
        ))
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
        .child(button("New video", Tone::Quiet, |s, tx| s.new_video(tx)))
}

pub fn view(s: &State, doc: &Document) -> El {
    let width = s.settings.reading_width.points();
    let count = doc.paragraphs.len() + 2;
    let list = col()
        .key("paragraphs")
        .w_full()
        .growth(GrowthSpec::default())
        .selection_group(s.reader.edits.is_none())
        .virtual_rows(
            count,
            s.settings.size as f32 * 6.5,
            move |s: &State, i| -> Cow<'_, str> {
                let n = s.doc.as_ref().map_or(0, |d| d.paragraphs.len());
                match i {
                    0 => "head".into(),
                    i if i > n => "end".into(),
                    i => format!("p-{}", i - 1).into(),
                }
            },
            |s: &State, i| {
                let doc = s.doc.as_ref().expect("reader has a transcript");
                match i {
                    0 => head(s, doc),
                    i if i > doc.paragraphs.len() => end(s),
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
                .pb(96.0)
                .child(
                    col()
                        .key("article-column")
                        .w_full()
                        .max_w(width)
                        .child(list),
                ),
        );
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
    pane.child(article).child(hud::pill(s))
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
    let style = style(s);
    let body = if let Some(edits) = &s.reader.edits {
        text_input(&edits[index], style)
            .key(format!("edit-{index}"))
            .label(format!("Paragraph {}", index + 1))
            .w_full()
            .min_h(style.line_px())
            .color(c(INK))
            .caret(c(INK), c(SELECTION))
            .on_edit(move |s: &mut State| &mut s.reader.edits.as_mut().unwrap()[index])
            .on_submit(move |s: &mut State, _| s.reader.edits.as_mut().unwrap()[index].insert("\n"))
    } else {
        text(&p.text)
            .font(style)
            .color(c(INK))
            .wrap()
            .selection_color(c(SELECTION))
            .selectable(true)
    };
    block.child(body)
}

fn end(s: &State) -> El {
    let mut end = col().w_full().gap(14.0).pt(8.0);
    if let Some(note) = hud::finish_note(s) {
        end = end.child(note);
    }
    end.child(
        row()
            .gap(10.0)
            .center_y()
            .child(
                button(
                    if s.reader.copied_at.is_some() {
                        "Copied"
                    } else {
                        "Copy transcript"
                    },
                    Tone::Quiet,
                    |s, tx| {
                        if let Some(all) = s.transcript_text() {
                            tx.copy(all);
                            s.reader.copied_at = Some(s.clock);
                        }
                    },
                )
                .key("copy")
                .label("Copy transcript"),
            )
            .child(button("Back to top", Tone::Ghost, |_, tx| {
                tx.scroll_to_with(ARTICLE, 0.0, 0.0, rune::Spring::new(0.6, 0.0));
            })),
    )
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
pub fn on_key(s: &mut State, e: &KeyEvent, tx: &mut Tx) -> bool {
    if s.reader.edits.is_some() {
        return false;
    }
    match &e.key {
        Key::Text(t) if t == " " => tx.scroll_page(ARTICLE, if e.mods.shift { -1 } else { 1 }),
        Key::Home => tx.scroll_to_with(ARTICLE, 0.0, 0.0, rune::Spring::new(0.6, 0.0)),
        Key::End => tx.scroll_to_with(ARTICLE, 0.0, f32::MAX, rune::Spring::new(0.6, 0.0)),
        _ => return false,
    }
    true
}
