use std::env::current_exe;
// src/openjtalk/dict.rs
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

const DICT_DIR_NAME: &str = "open_jtalk_dic_utf_8-1.11";
const DICT_URL: &str =
    "https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz";

/// 辞書ディレクトリのパスを解決する。
/// 1. OPEN_JTALK_DICT_DIR が存在すればそれを使う
/// 2. なければ <クレートディレクトリ>/open_jtalk_dic_utf_8-1.11/ を使う(無ければダウンロード)
pub fn resolve_dict_dir() -> Result<PathBuf, String> {
    if let Ok(env_dir) = std::env::var("OPEN_JTALK_DICT_DIR") {
        let path = PathBuf::from(&env_dir);
        if path.is_dir() {
            return Ok(path);
        }
        eprintln!(
            "warning: OPEN_JTALK_DICT_DIR={} is set but not a valid directory. Falling back.",
            env_dir
        );
    }

    let default_dir = default_dict_dir();
    if default_dir.is_dir() && dir_has_dict_files(&default_dir) {
        return Ok(default_dir);
    }

    download_and_extract(&default_dir)?;
    Ok(default_dir)
}

fn default_dict_dir() -> PathBuf {
    // 実行ファイルがあるディレクトリを起点にする
    current_exe().unwrap().parent().unwrap().join(DICT_DIR_NAME)
}

fn dir_has_dict_files(dir: &Path) -> bool {
    // 辞書の主要ファイルが揃っているかを簡易チェック
    dir.join("sys.dic").is_file() && dir.join("matrix.bin").is_file()
}

fn download_and_extract(target_dir: &Path) -> Result<(), String> {
    eprintln!(
        "OpenJTalk dictionary not found. Downloading from {}...",
        DICT_URL
    );

    // 辞書をダウンロード
    let mut bytes = Vec::new();
    let response = ureq::get(DICT_URL)
        .call()
        .map_err(|e| format!("Failed to download dictionary: {e}"))?;
    response
        .into_parts()
        .1
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Failed to read response body: {e}"))?;

    // 親ディレクトリの作成
    let parent = target_dir
        .parent()
        .ok_or_else(|| "Invalid target directory (no parent)".to_string())?;
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("Failed to create parent directory: {e}"))?;

    // ダウンロードしたtar.gzファイルを展開
    let tar = flate2::read::GzDecoder::new(Cursor::new(bytes));
    let mut archive = tar::Archive::new(tar);
    archive
        .unpack(parent)
        .map_err(|e| format!("Failed to extract dictionary archive: {e}"))?;

    if !dir_has_dict_files(target_dir) {
        return Err(format!(
            "Dictionary extraction succeeded but expected files not found in {:?}",
            target_dir
        ));
    }

    eprintln!("Dictionary extracted to {:?}", target_dir);
    Ok(())
}
