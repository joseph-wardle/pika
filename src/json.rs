//! The JSON `slangc -reflection-json` writes: objects, lists, strings, numbers and the three
//! keywords, with no escapes beyond `\"` and `\\`.

#[derive(Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    Text(String),
    List(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    /// An object's member, or none for a missing key or a non-object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Self::Object(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text(t) => Some(t),
            _ => None,
        }
    }

    pub fn number(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// A list's items; empty for anything else.
    pub fn list(&self) -> &[Json] {
        match self {
            Self::List(items) => items,
            _ => &[],
        }
    }
}

pub fn parse(text: &str) -> Result<Json, String> {
    let mut p = Parser {
        bytes: text.as_bytes(),
        at: 0,
    };
    let value = p.value()?;
    p.skip_space();
    if p.at != p.bytes.len() {
        return Err(p.error("text after the value"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn error(&self, what: &str) -> String {
        format!("invalid JSON at byte {}: {what}", self.at)
    }

    fn skip_space(&mut self) {
        while self.at < self.bytes.len() && self.bytes[self.at].is_ascii_whitespace() {
            self.at += 1;
        }
    }

    fn take(&mut self, byte: u8) -> bool {
        self.skip_space();
        if self.bytes.get(self.at) == Some(&byte) {
            self.at += 1;
            return true;
        }
        false
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        if self.take(byte) {
            return Ok(());
        }
        Err(self.error(&format!("expected '{}'", byte as char)))
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_space();
        match self.bytes.get(self.at) {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(Json::Text),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => self.keyword(),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut members = Vec::new();
        if self.take(b'}') {
            return Ok(Json::Object(members));
        }
        loop {
            self.skip_space();
            let key = self.string()?;
            self.expect(b':')?;
            members.push((key, self.value()?));
            if self.take(b',') {
                continue;
            }
            self.expect(b'}')?;
            return Ok(Json::Object(members));
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        if self.take(b']') {
            return Ok(Json::List(items));
        }
        loop {
            items.push(self.value()?);
            if self.take(b',') {
                continue;
            }
            self.expect(b']')?;
            return Ok(Json::List(items));
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut text = Vec::new();
        loop {
            let byte = *self
                .bytes
                .get(self.at)
                .ok_or(self.error("unterminated string"))?;
            self.at += 1;
            match byte {
                b'"' => break,
                b'\\' => {
                    let escaped = *self
                        .bytes
                        .get(self.at)
                        .ok_or(self.error("unterminated string"))?;
                    self.at += 1;
                    match escaped {
                        b'"' | b'\\' | b'/' => text.push(escaped),
                        b'n' => text.push(b'\n'),
                        b't' => text.push(b'\t'),
                        _ => return Err(self.error("unsupported escape")),
                    }
                }
                _ => text.push(byte),
            }
        }
        String::from_utf8(text).map_err(|_| self.error("invalid UTF-8"))
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        while self.at < self.bytes.len()
            && matches!(
                self.bytes[self.at],
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'
            )
        {
            self.at += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).unwrap();
        text.parse()
            .map(Json::Number)
            .map_err(|_| self.error("invalid number"))
    }

    fn keyword(&mut self) -> Result<Json, String> {
        for (word, value) in [
            ("null", Json::Null),
            ("true", Json::Bool(true)),
            ("false", Json::Bool(false)),
        ] {
            if self.bytes[self.at..].starts_with(word.as_bytes()) {
                self.at += word.len();
                return Ok(value);
            }
        }
        Err(self.error("expected a value"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_reflection_shaped_document() {
        let doc = parse(
            r#"{"parameters": [{"name": "pixelCB", "binding": {"kind": "descriptorTableSlot", "index": 0, "count": 3}}], "x": [true, false, null, -1.5e2, "a\"b"]}"#,
        )
        .unwrap();
        let first = &doc.get("parameters").unwrap().list()[0];
        assert_eq!(first.get("name").unwrap().text(), Some("pixelCB"));
        assert_eq!(
            first.get("binding").unwrap().get("count").unwrap().number(),
            Some(3.0)
        );
        assert_eq!(
            doc.get("x").unwrap().list(),
            [
                Json::Bool(true),
                Json::Bool(false),
                Json::Null,
                Json::Number(-150.0),
                Json::Text("a\"b".into())
            ]
        );
        assert_eq!(doc.get("missing"), None);
        assert_eq!(parse(" {} ").unwrap(), Json::Object(vec![]));
    }

    #[test]
    fn rejects_broken_documents() {
        for text in ["", "{", "[1,]", "{\"a\" 1}", "{} x", "\"open", "tru"] {
            assert!(parse(text).is_err(), "{text:?}");
        }
    }
}
