use super::*;

#[test]
fn reader_corrects_text_and_remembers_width_and_zoom() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    let mut s = load_state(Some(path.clone()));
    s.settings.configured = true;
    s.open_reader(Document {
        title: "Interview".into(),
        channel: String::new(),
        link: "https://www.youtube.com/watch?v=jNQXAC9IVRw".into(),
        source: "Captions".into(),
        language: "en".into(),
        speaker_note: String::new(),
        paragraphs: vec![Paragraph {
            text: "I read the wrong word.".into(),
            speaker: Some("Speaker 3".into()),
            start: 23.0,
        }],
    });
    let mut rt = Runtime::new(s, ui::view, (1200.0, 820.0), Env::default());
    rt.frame(0.0);
    tap(&mut rt, "Aa");
    tap(&mut rt, "Wide");
    tap(&mut rt, "Aa");
    let column = rt
        .elements()
        .into_iter()
        .find(|e| e.key.as_deref() == Some("article-column"))
        .unwrap();
    assert!(
        rt.state.settings.reading_width == ReadingWidth::Wide,
        "Wide was not selected"
    );
    assert!(
        (column.target[2] - 900.0).abs() < 1.0,
        "Column bounds {:?}",
        column.target
    );
    let (x, y) = rt.find("p-0").unwrap();
    rt.wheel_event(rune::WheelEvent::new(0.0, 40.0).at(x, y).mods(rune::Mods {
        ctrl: true,
        ..Default::default()
    }));
    rt.frame(0.02);
    assert_eq!(rt.state.settings.size, 22);
    rt.wheel_event(rune::WheelEvent::new(0.0, 40.0).at(x, y));
    rt.frame(0.02);
    assert_eq!(rt.state.settings.size, 22);
    tap(&mut rt, "Edit");
    tap(&mut rt, "edit-0");
    rt.key(rune::KeyEvent {
        key: rune::Key::Command('a'),
        mods: rune::Mods {
            cmd: true,
            ..Default::default()
        },
    });
    rt.frame(0.02);
    rt.paste("I read the right word.");
    rt.frame(0.02);
    tap(&mut rt, "Done");
    let p = &rt.state.doc.as_ref().unwrap().paragraphs[0];
    assert_eq!(p.text, "I read the right word.");
    assert_eq!(p.speaker.as_deref(), Some("Speaker 3"));
    assert_eq!(p.start, 23.0);
    drop(rt);
    let saved = load_state(Some(path));
    assert!(saved.settings.reading_width == ReadingWidth::Wide);
    assert_eq!(saved.settings.size, 22);
}
