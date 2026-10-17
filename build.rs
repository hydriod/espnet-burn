fn main() {
    let mut build = cxx_build::bridge("src/openjtalk/mod.rs");
    build
        .include("lib")
        .include("lib/open_jtalk/src")
        .include("lib/open_jtalk/src/mecab/src")
        .include("lib/open_jtalk/src/jpcommon")
        .include("lib/open_jtalk/src/njd")
        .include("lib/open_jtalk/src/njd2jpcommon")
        .include("lib/open_jtalk/src/mecab2njd")
        .include("lib/open_jtalk/src/text2mecab")
        .include("lib/open_jtalk/src/njd_set_accent_phrase")
        .include("lib/open_jtalk/src/njd_set_accent_type")
        .include("lib/open_jtalk/src/njd_set_digit")
        .include("lib/open_jtalk/src/njd_set_long_vowel")
        .include("lib/open_jtalk/src/njd_set_pronunciation")
        .include("lib/open_jtalk/src/njd_set_unvoiced_vowel")
        .define("CHARSET_UTF_8", "1") // 辞書がUTF-8版ならこのまま。Shift-JIS辞書ならCHARSET_SHIFT_JISに変更
        .define("DIC_VERSION", "102")
        .define("MECAB_WITHOUT_MUTEX_LOCK", None)
        .define("MECAB_DEFAULT_RC", "\"dummy\"")
        .define("MECAB_UTF8_USE_ONLY", None)
        .define("PACKAGE", "\"open_jtalk\"")
        .define("VERSION", "\"1.11\"");

    // OS固有の定義:
    // Windows(MSVC)では windows.h が存在する前提だったが、
    // Linuxでは HAVE_WINDOWS_H を定義すると <windows.h> が無くて失敗する。
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        build.define("HAVE_WINDOWS_H", None);
    } else {
        // CMakeのLinuxビルド相当 (HAVE_CONFIG_Hの代わりに個別定義)
        for macro_name in [
            "HAVE_CTYPE_H",
            "HAVE_DIRENT_H",
            "HAVE_FCNTL_H",
            "HAVE_GETENV",
            "HAVE_GETPAGESIZE",
            "HAVE_INTTYPES_H",
            "HAVE_MEMORY_H",
            "HAVE_MMAP",
            "HAVE_OPENDIR",
            "HAVE_SETJMP",
            "HAVE_SETJMP_H",
            "HAVE_SQRT",
            "HAVE_STDINT_H",
            "HAVE_STDLIB_H",
            "HAVE_STRINGS_H",
            "HAVE_STRING_H",
            "HAVE_STRSTR",
            "HAVE_SYS_MMAN_H",
            "HAVE_SYS_STAT_H",
            "HAVE_SYS_TYPES_H",
            "HAVE_UNISTD_H",
            "HAVE_PTHREAD_H",
        ] {
            build.define(macro_name, "1");
        }
    }

    build
        .file("lib/openjtalk_bridge.cpp")
        .file("lib/open_jtalk/src/text2mecab/text2mecab.c")
        // --- MeCab本体（学習・辞書生成ツール群は除外） ---
        .file("lib/open_jtalk/src/mecab/src/char_property.cpp")
        .file("lib/open_jtalk/src/mecab/src/connector.cpp")
        .file("lib/open_jtalk/src/mecab/src/context_id.cpp")
        .file("lib/open_jtalk/src/mecab/src/dictionary.cpp")
        .file("lib/open_jtalk/src/mecab/src/dictionary_rewriter.cpp")
        .file("lib/open_jtalk/src/mecab/src/feature_index.cpp")
        .file("lib/open_jtalk/src/mecab/src/iconv_utils.cpp")
        .file("lib/open_jtalk/src/mecab/src/libmecab.cpp")
        .file("lib/open_jtalk/src/mecab/src/mecab.cpp")
        .file("lib/open_jtalk/src/mecab/src/nbest_generator.cpp")
        .file("lib/open_jtalk/src/mecab/src/param.cpp")
        .file("lib/open_jtalk/src/mecab/src/string_buffer.cpp")
        .file("lib/open_jtalk/src/mecab/src/tagger.cpp")
        .file("lib/open_jtalk/src/mecab/src/tokenizer.cpp")
        .file("lib/open_jtalk/src/mecab/src/utils.cpp")
        .file("lib/open_jtalk/src/mecab/src/viterbi.cpp")
        .file("lib/open_jtalk/src/mecab/src/writer.cpp")
        // --- ここまで ---
        .file("lib/open_jtalk/src/njd/njd.c")
        .file("lib/open_jtalk/src/njd/njd_node.c")
        .file("lib/open_jtalk/src/njd2jpcommon/njd2jpcommon.c")
        .file("lib/open_jtalk/src/njd_set_accent_phrase/njd_set_accent_phrase.c")
        .file("lib/open_jtalk/src/njd_set_accent_type/njd_set_accent_type.c")
        .file("lib/open_jtalk/src/njd_set_digit/njd_set_digit.c")
        .file("lib/open_jtalk/src/njd_set_long_vowel/njd_set_long_vowel.c")
        .file("lib/open_jtalk/src/njd_set_pronunciation/njd_set_pronunciation.c")
        .file("lib/open_jtalk/src/njd_set_unvoiced_vowel/njd_set_unvoiced_vowel.c")
        .file("lib/open_jtalk/src/mecab2njd/mecab2njd.c")
        .file("lib/open_jtalk/src/jpcommon/jpcommon.c")
        .file("lib/open_jtalk/src/jpcommon/jpcommon_label.c")
        .file("lib/open_jtalk/src/jpcommon/jpcommon_node.c")
        .flag_if_supported("/std:c++11")
        .flag_if_supported("-std=c++11")
        .flag_if_supported("/utf-8")
        // GCCで .c をC++としてコンパイルすると 0xC0 などが narrowing error になる対策
        .flag_if_supported("-Wno-narrowing")
        .flag_if_supported("-funsigned-char")
        .flag_if_supported("-Wno-write-strings")
        .compile("espnet_burn");

    println!("cargo:rerun-if-changed=lib/open_jtalk/src");
    println!("cargo:rerun-if-changed=lib/openjtalk_bridge.cpp");
    println!("cargo:rerun-if-changed=lib/openjtalk_bridge.h");
}
