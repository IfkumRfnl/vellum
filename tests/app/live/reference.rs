//! Published transcripts that name who said each line, to score speaker labels against.

fn fetch(url: &str) -> String {
    ureq::get(url)
        .set("User-Agent", "Mozilla/5.0 (Vellum live tests)")
        .call()
        .unwrap_or_else(|e| panic!("{url}: {e}"))
        .into_string()
        .unwrap_or_else(|e| panic!("{url}: {e}"))
}

/// Markup as plain text: tags become spaces, entities characters.
fn plain(html: &str) -> String {
    let mut text = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => {
                in_tag = false;
                text.push(' ');
            }
            _ if !in_tag => text.push(c),
            _ => {}
        }
    }
    let text = html_escape::decode_html_entities(&text);
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A BBC Learning English episode's script: each line under its speaker's bold name. Vocabulary
/// is bold too, mid-sentence; a name stands alone before a line break.
pub fn bbc_script(url: &str) -> Vec<(String, String)> {
    let page = fetch(url);
    let start = page
        .find(">TRANSCRIPT<")
        .expect("BBC page without a transcript");
    let end = page[start..]
        .find("</div>")
        .map_or(page.len(), |e| start + e);
    let mut rest = &page[start..end];
    let mut turns: Vec<(String, String)> = Vec::new();
    let mut name: Option<String> = None;
    let mut text = String::new();
    while let Some(at) = rest.find("<strong>") {
        text.push_str(&rest[..at]);
        let after = &rest[at + "<strong>".len()..];
        let close = after.find("</strong>").expect("unclosed <strong>");
        let (inner, tail) = (&after[..close], &after[close + "</strong>".len()..]);
        let label = plain(inner);
        let alone = inner.trim_end().ends_with("/>") || tail.trim_start().starts_with("<br");
        if alone && !label.is_empty() && label.chars().count() <= 40 {
            if let Some(name) = name.take() {
                turns.push((name, plain(&text)));
            }
            name = Some(label);
            text.clear();
        } else {
            text.push_str(inner);
        }
        rest = tail;
    }
    text.push_str(rest);
    turns.extend(name.map(|name| (name, plain(&text))));
    assert!(
        turns.len() >= 20,
        "{url}: only {} script lines",
        turns.len()
    );
    turns
}

/// A Lex Fridman Podcast transcript: segments each with the speaker's name.
pub fn lex_transcript(url: &str) -> Vec<(String, String)> {
    let page = fetch(url);
    let mut turns: Vec<(String, String)> = Vec::new();
    for segment in page.split("<div class=\"ts-segment\">").skip(1) {
        let segment = &segment[..segment.find("</div>").unwrap_or(segment.len())];
        let field = |class: &str| {
            let open = format!("<span class=\"{class}\">");
            segment.find(&open).map(|at| {
                let body = &segment[at + open.len()..];
                plain(&body[..body.find("</span>").unwrap_or(body.len())])
            })
        };
        let text = field("ts-text").unwrap_or_default();
        let name = field("ts-name")
            .filter(|n| !n.is_empty())
            .or_else(|| turns.last().map(|t| t.0.clone()))
            .expect("transcript starts without a speaker");
        turns.push((name, text));
    }
    assert!(turns.len() >= 100, "{url}: only {} segments", turns.len());
    turns
}
