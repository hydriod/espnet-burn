use burn::tensor::DataError;

#[derive(thiserror::Error, Debug, Clone)]
pub enum Error {
    #[error("argument error: {0}")]
    Argument(String),

    #[error("IO error: {0}")]
    Io(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("RegEx error: {0}")]
    Regex(#[from] regex::Error),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("request error: {0}")]
    Request(String),

    #[error("other error: {0}")]
    Other(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

impl From<yaml_serde::Error> for Error {
    fn from(e: yaml_serde::Error) -> Self {
        Error::Parse(e.to_string())
    }
}

impl From<ureq::Error> for Error {
    fn from(e: ureq::Error) -> Self {
        Error::Request(e.to_string())
    }
}
