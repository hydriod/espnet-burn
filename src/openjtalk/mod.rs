mod dict;

use std::sync::OnceLock;

use crate::openjtalk::dict::resolve_dict_dir;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("openjtalk_bridge.h");

        fn run_frontend(text: &str, use_unvoiced_vowel: bool, output: &mut Vec<String>);
        // fn make_label(text: &str, output: &mut Vec<String>);
        fn init_dictionary(dict_dir_str: &str) -> bool;
    }
}

static INIT: OnceLock<Result<(), String>> = OnceLock::new();

fn ensure_dict_loaded() -> Result<(), String> {
    INIT.get_or_init(|| {
        let dict_dir = resolve_dict_dir()?;
        let dict_dir_str = dict_dir
            .to_str()
            .ok_or_else(|| "Dictionary path is not valid UTF-8".to_string())?;
        if ffi::init_dictionary(dict_dir_str) {
            Ok(())
        } else {
            Err(format!("Failed to load dictionary from {:?}", dict_dir))
        }
    })
    .clone()
}

/// Converts Japanese text to HTS-format labels using OpenJTalk.
///
/// This is equivalent to pyopenjtalk.make_label() but calls OpenJTalk C++ directly.
///
/// # Arguments
/// * `text` - Input Japanese text (UTF-8)
///
/// # Returns
/// Vec of label strings in HTS Full Context Label format
///
/// # Example
/// ```ignore
/// let labels = make_label("こんにちは").unwrap();
/// for label in labels {
///     println!("{}", label);
/// }
/// ```
///
/// # Errors
/// Returns error if text processing or MeCab analysis fails
/*
pub fn make_label(text: &str) -> Result<Vec<String>, String> {
    let mut output = Vec::new();
    ffi::make_label(text, &mut output);
    if output.is_empty() {
        Err("Failed to process text or no features generated".to_string())
    } else {
        Ok(output)
    }
}
*/

/// Runs the full OpenJTalk frontend with control over unvoiced vowel handling.
///
/// This is equivalent to pyopenjtalk.run_frontend() but calls OpenJTalk C++ directly.
/// This function gives more control than make_label() by allowing unvoiced vowel preservation.
///
/// # Arguments
/// * `text` - Input Japanese text (UTF-8)
/// * `use_unvoiced_vowel` - If true, preserves unvoiced vowels (A/I/U/E/O) as-is.
///                          If false, converts them to lowercase (a/i/u/e/o).
///
/// # Returns
/// Vec of label strings in HTS Full Context Label format
///
/// # Example
/// ```ignore
/// let labels = run_frontend("おはようございます", false).unwrap();
/// ```
///
/// # Errors
/// Returns error if text processing or MeCab analysis fails
pub fn run_frontend(text: &str, use_unvoiced_vowel: bool) -> Result<Vec<String>, String> {
    ensure_dict_loaded()?;
    let mut output = Vec::new();

    ffi::run_frontend(text, use_unvoiced_vowel, &mut output);
    if output.is_empty() {
        Err("Failed to process text or no features generated".to_string())
    } else {
        Ok(output)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_run_frontend_with_unvoiced_vowel() {
        let result = run_frontend("おはよう", true);
        println!("{}", result.clone().err().unwrap_or("".to_string()));
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_frontend_without_unvoiced_vowel() {
        let result = run_frontend("あ", false);
        println!("{}", result.clone().err().unwrap_or("".to_string()));
        assert!(result.is_ok());
    }

    /*
    #[test]
    fn test_make_label_simple() {
        let result = make_label("あ");
        assert!(result.is_ok());
        let labels = result.unwrap();
        assert!(!labels.is_empty());
        println!("{}", labels.join(" "));
    }

    #[test]
    fn test_make_label_complex() {
        let result = make_label("こんにちは");
        assert!(result.is_ok());
        let labels = result.unwrap();
        assert!(labels.len() > 0);
        for label in &labels {
            assert!(!label.is_empty());
        }
    }
    */
}
