use crate::{error::Error, openjtalk::*};
use regex::Regex;

/// Converts Japanese text into phonemes with prosody markers.
///
/// ```
/// use espnet_burn::text::phoneme_tokenizer::pyopenjtalk_g2p_prosody;
///
/// assert_eq!(
///     pyopenjtalk_g2p_prosody("こんにちは", true).unwrap(),
///     vec!["^", "k", "o", "[", "N", "n", "i", "ch", "i", "w", "a", "$"]
///         .into_iter()
///         .map(String::from)
///         .collect::<Vec<_>>()
/// );
/// ```
pub fn pyopenjtalk_g2p_prosody(
    text: &str,
    drop_unvoiced_vowel: bool,
) -> Result<Vec<String>, Error> {
    let label = extract_fullcontext_label(text)?;

    let mut phones = vec![];
    for i in 0..label.len() {
        let lab_curr = &label[i];

        // current phoneme
        let p3 = Regex::new(r"\-(.*?)\+")?
            .captures(lab_curr)
            .ok_or_else(|| {
                Error::Regex(regex::Error::Syntax(
                    "Failed to parse phoneme label".to_string(),
                ))
            })?;
        let mut p3 = p3[1].to_string();

        // drop unvoiced vowel as normal vowels
        if drop_unvoiced_vowel && (p3 == "A" || p3 == "I" || p3 == "U" || p3 == "E" || p3 == "O") {
            p3 = p3.to_lowercase();
        }

        // deal with sil at the beginning and end of the text
        if p3 == "sil" {
            if !(i == 0 || i == label.len() - 1) {
                return Err(Error::Tokenizer(
                    "Unexpected sil token position".to_string(),
                ));
            }
            if i == 0 {
                phones.push("^".to_string());
            } else if i == label.len() - 1 {
                // check question form or not
                let e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr);
                if e3 == 0 {
                    phones.push("$".to_string());
                } else if e3 == 1 {
                    phones.push("?".to_string());
                }
            }
            continue;
        } else if p3 == "pau" {
            phones.push("_".to_string());
            continue;
        } else {
            phones.push(p3.clone());
        }

        // accent type and position info (forward or backward)
        let a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr);
        let a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr);
        let a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr);

        // number of mora in accent phrase
        let f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr);

        let a2_next = if i + 1 < label.len() {
            _numeric_feature_by_regex(r"\+(\d+)\+", &label[i + 1])
        } else {
            -50
        };

        // accent phrase border
        if a3 == 1 && a2_next == 1 {
            phones.push("#".to_string());
        } else if a1 == 0 && a2_next == a2 + 1 && a2 != f1 {
            // pitch falling
            phones.push("]".to_string());
        } else if a2 == 1 && a2_next == 2 {
            // pitch rising
            phones.push("[".to_string());
        }
    }

    Ok(phones)
}

fn _numeric_feature_by_regex(regex: &str, s: &str) -> i128 {
    if let Some(captures) = Regex::new(regex).unwrap().captures(s) {
        captures[1].parse::<i128>().unwrap_or(-50)
    } else {
        -50
    }
}

fn extract_fullcontext_label(text: &str) -> Result<Vec<String>, Error> {
    run_frontend(text, false)
    // Ok(iter.map(|s| s.to_string()).collect::<Vec<String>>())
}
