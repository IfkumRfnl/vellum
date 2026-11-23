use super::*;

#[test]
fn continuing_uses_speech_and_skipping_can_be_reversed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    let mut rt = Runtime::new(
        load_state(Some(path.clone())),
        ui::view,
        (540.0, 560.0),
        Env {
            system_dark: true,
            scale: 1.0,
        },
    );
    rt.frame(0.0);
    assert!(rt.find("captions").is_none());
    assert!(rt.find("Label speakers").is_none());
    tap(&mut rt, "Continue");
    assert!(matches!(rt.state.screen, Screen::Home));
    assert_eq!(rt.state.settings.model, "parakeet_unified_en");
    assert!(!rt.state.settings.captions_first);
    tap(&mut rt, "Label speakers");
    assert!(rt.state.settings.speakers);
    tap(&mut rt, "url");
    rt.paste("https://youtu.be/jNQXAC9IVRw");
    rt.frame(0.02);
    tap(&mut rt, "Transcription");
    assert!(matches!(rt.state.screen, Screen::Home));
    assert!(rt.find("model-picker").is_none());
    tap(&mut rt, "captions");
    assert_eq!(rt.state.settings.model, "captions");
    assert!(rt.state.settings.speakers);
    assert!(rt.find("Label speakers").is_some());
    tap(&mut rt, "Transcription");
    assert_eq!(rt.state.settings.model, "captions");
    rt.key(rune::KeyEvent {
        key: rune::Key::Escape,
        mods: Default::default(),
    });
    rt.frame(0.02);
    assert!(!rt.state.sources_open);
    assert_eq!(rt.state.settings.model, "captions");
    tap(&mut rt, "Transcription");
    tap(&mut rt, "turbo");
    assert_eq!(rt.state.settings.model, "turbo");
    assert!(!rt.state.sources_open);
    assert_eq!(rt.state.input.text, "https://youtu.be/jNQXAC9IVRw");
    assert!(matches!(rt.state.screen, Screen::Home));
    drop(rt);
    let reopened = load_state(Some(path));
    assert_eq!(reopened.settings.model, "turbo");
    assert!(reopened.settings.speakers);
}

#[test]
#[ignore = "prepares Whisper Small (270 MB on a cold cache)"]
fn model_preparation_continues_after_setup_and_is_recognized_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    let runtime = || {
        let mut s = load_state(Some(path.clone()));
        s.wake = Some(Arc::new(|| {}));
        let mut rt = Runtime::new(
            s,
            ui::view,
            (540.0, 560.0),
            Env {
                system_dark: true,
                scale: 1.0,
            },
        );
        rt.poll = Some(poll);
        rt.frame(0.0);
        rt
    };
    let wait_for_model = |rt: &mut Runtime<State>| {
        let deadline = Instant::now() + Duration::from_secs(300);
        while rt.state.model_job.is_some() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(80));
            rt.frame(0.08);
        }
        assert!(
            rt.state.ready_models.iter().any(|model| model == "small"),
            "{}",
            rt.state.message
        );
    };
    let mut rt = runtime();
    tap(&mut rt, "small");
    assert!(matches!(&rt.state.model_job, Some(m) if m.id == "small"));
    tap(&mut rt, "Continue");
    assert!(matches!(rt.state.screen, Screen::Home));
    wait_for_model(&mut rt);
    drop(rt);
    let mut rt = runtime();
    tap(&mut rt, "Transcription");
    wait_for_model(&mut rt);
    assert_eq!(rt.state.settings.model, "small");
    tap(&mut rt, "small");
    assert!(rt.state.model_job.is_none());
}
