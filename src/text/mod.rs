pub mod phoneme_tokenizer;
pub mod token_id_converter;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_text2ids() {
        let text = "こんにちは";
        let converter = token_id_converter::TokenIDConverter::new(
            vec![
                "<blank>",
                "<unk>",
                "a",
                "o",
                "i",
                "[",
                "#",
                "]",
                "u",
                "e",
                "t",
                "k",
                "n",
                "r",
                "m",
                "N",
                "sh",
                "s",
                "d",
                "g",
                "_",
                "^",
                "$",
                "cl",
                "w",
                "h",
                "b",
                "ch",
                "y",
                "ts",
                "j",
                "z",
                "p",
                "f",
                "ky",
                "gy",
                "ry",
                "ny",
                "hy",
                "my",
                "by",
                "py",
                "?",
                "dy",
                "v",
                "ty",
                "<sos/eos>",
            ],
            Some("<unk>".to_string()),
        )
        .unwrap();
        let tokens = phoneme_tokenizer::pyopenjtalk_g2p_prosody(text, true).unwrap();
        let ids = converter.tokens2ids(&tokens);

        assert_eq!(ids, vec![21, 11, 3, 5, 15, 12, 4, 27, 4, 24, 2, 22]);
    }
}
