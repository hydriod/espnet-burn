use crate::error::Error;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct TokenIDConverter {
    pub token_list: Vec<String>,
    pub token_list_repr: String,
    pub token2id: HashMap<String, usize>,
    pub unk_symbol: String,
    pub unk_id: usize,
}

impl TokenIDConverter {
    pub fn new<I, S>(token_list: I, unk_symbol: Option<String>) -> Result<Self, Error>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let token_list_vec: Vec<String> = token_list
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();

        // Build representation
        let mut token_list_repr = String::new();
        let nvocab = token_list_vec.len();
        for (i, t) in token_list_vec.iter().enumerate() {
            if i == 3 {
                break;
            }
            token_list_repr.push_str(t);
            token_list_repr.push_str(", ");
        }
        token_list_repr.push_str(&format!("... (NVocab={nvocab})"));

        let mut token2id: HashMap<String, usize> = HashMap::with_capacity(token_list_vec.len());
        for (i, t) in token_list_vec.iter().enumerate() {
            if token2id.contains_key(t) {
                return Err(Error::Argument(format!(r#"Symbol "{t}" is duplicated"#)));
            }
            token2id.insert(t.clone(), i);
        }

        let unk_symbol = unk_symbol.unwrap_or_else(|| "<unk>".to_string());
        let unk_id = token2id.get(&unk_symbol).copied().ok_or_else(|| {
            Error::Argument(format!(
                "Unknown symbol '{unk_symbol}' doesn't exist in the token_list"
            ))
        })?;

        Ok(Self {
            token_list: token_list_vec,
            token_list_repr,
            token2id,
            unk_symbol,
            unk_id,
        })
    }

    pub fn from_file<P: AsRef<Path>>(
        token_list_path: P,
        unk_symbol: Option<String>,
    ) -> Result<Self, Error> {
        let path = token_list_path.as_ref();
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut token_list_vec: Vec<String> = Vec::new();

        for (_idx, line) in reader.lines().enumerate() {
            let line = line?;
            // Python: line[0] + line[1:].rstrip()
            if line.is_empty() {
                token_list_vec.push(String::new());
                continue;
            }
            let chars: Vec<char> = line.chars().collect();
            // rstrip rest after first char? Python does: line[0] + line[1:].rstrip()
            if chars.len() > 1 {
                let mut end = chars.len() - 1;
                while end > 0
                    && (chars[end] == '\r'
                        || chars[end] == '\n'
                        || chars[end] == ' '
                        || chars[end] == '\t')
                {
                    end -= 1;
                }
                let mut new_chars = vec![chars[0]];
                for c in chars[1..=end].iter() {
                    new_chars.push(*c);
                }
                token_list_vec.push(new_chars.into_iter().collect());
            } else {
                token_list_vec.push(
                    line.trim_end_matches(|c: char| c.is_whitespace())
                        .to_string(),
                );
            }
        }

        let path_str = path.to_string_lossy().to_string();
        // Rebuild properly
        let mut converter = TokenIDConverter {
            token_list: token_list_vec.clone(),
            token_list_repr: path_str.clone(),
            token2id: HashMap::new(),
            unk_symbol: unk_symbol.clone().unwrap_or_else(|| "<unk>".to_string()),
            unk_id: 0,
        };

        // Rebuild token2id with the logic
        converter.token2id.clear();
        for (i, t) in converter.token_list.iter().enumerate() {
            if converter.token2id.contains_key(t) {
                return Err(Error::Argument(format!(r#"Symbol "{t}" is duplicated"#)));
            }
            converter.token2id.insert(t.clone(), i);
        }

        let unk_sym = unk_symbol.unwrap_or_else(|| "<unk>".to_string());
        converter.unk_symbol = unk_sym.clone();
        converter.unk_id = converter.token2id.get(&unk_sym).copied().ok_or_else(|| {
            Error::Argument(format!(
                "Unknown symbol '{unk_sym}' doesn't exist in the token_list"
            ))
        })?;

        Ok(converter)
    }

    pub fn get_num_vocabulary_size(&self) -> usize {
        self.token_list.len()
    }

    pub fn ids2tokens(&self, integers: &[usize]) -> Vec<String> {
        integers
            .iter()
            .map(|&i| self.token_list[i].clone())
            .collect()
    }

    pub fn ids2tokens_iter<I>(&self, integers: I) -> Vec<String>
    where
        I: IntoIterator<Item = usize>,
    {
        integers
            .into_iter()
            .map(|i| self.token_list[i].clone())
            .collect()
    }

    pub fn tokens2ids<I, S>(&self, tokens: I) -> Vec<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        tokens
            .into_iter()
            .map(|t| {
                let s = t.as_ref();
                *self.token2id.get(s).unwrap_or(&self.unk_id)
            })
            .collect()
    }
}
