//! The **Aa** panel: typeface, size, width, spacing and timestamps.

use crate::{
    State,
    settings::ReadingWidth,
    ui::{
        theme::{EDGE, INK, PANEL, c, figures, kit},
        widgets::{Tone, button, choice, label, toggle},
    },
};
use rune::{El, col, row};

pub fn panel(s: &State) -> El {
    let ty = &kit().ty;
    let mut fonts = row().gap(6.0);
    for (i, name) in ["Serif", "Sans", "Mono"].into_iter().enumerate() {
        fonts = fonts.child(choice(name, s.settings.font == i, move |s, _| {
            s.settings.font = i;
            s.save();
        }));
    }
    let size = row()
        .gap(8.0)
        .center_y()
        .child(button("−", Tone::Quiet, |s, _| s.adjust_font(-1)).label("Smaller type"))
        .child(
            label(format!("{} pt", s.settings.size), figures(13.0, 500.0), INK)
                .w(44.0)
                .align(rune::Align::Center),
        )
        .child(button("+", Tone::Quiet, |s, _| s.adjust_font(1)).label("Larger type"));
    let mut widths = row().gap(6.0).role("radiogroup").label("Reading width");
    for (name, width) in ReadingWidth::ALL {
        let on = s.settings.reading_width == width;
        widths = widths.child(
            choice(name, on, move |s, _| {
                s.settings.reading_width = width;
                s.save();
            })
            .role("radio")
            .checked(on),
        );
    }
    col()
        .key("appearance")
        .gap(10.0)
        .padding(14.0)
        .w_full()
        .fill(c(PANEL))
        .radius(12.0)
        .border(1.0, c(EDGE))
        .shadow(10.0, 30.0, rune::Color::BLACK.alpha(0.45))
        .child(row().justify_between().center_y().child(fonts).child(size))
        .child(
            row()
                .justify_between()
                .center_y()
                .child(label("Reading width", ty.ui, INK))
                .child(widths),
        )
        .child(toggle("Relaxed line spacing", s.settings.spacious, |s| {
            s.settings.spacious = !s.settings.spacious;
            s.save();
        }))
        .child(toggle("Timestamps", s.settings.timestamps, |s| {
            s.settings.timestamps = !s.settings.timestamps;
            s.save();
        }))
}
