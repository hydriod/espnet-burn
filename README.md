# espnet-burn

ESPnet2 の音声合成（TTS）モデル、特に **VITS** の推論処理を
[Burn](https://burn.dev) フレームワークへ移植した Rust ライブラリです。

日本語テキストの解析・音素化には **Open JTalk** を CXX FFI 経由で直接利用し、
学習済みの PyTorch チェックポイント（`.pth`）を `burn-store` でそのままロードします。
最終的には「テキスト入力から音声波形生成まで」を Rust のみで完結させることを目指しています。

> **移植対象は COEIROINK です。** 本ライブラリは COEIROINK の推論用途を想定して移植を進めて
> いるため、COEIROINK が使用しない機能（話者・言語 conditioning、teacher forcing、
> pitch / energy 系など）は意図的に未実装のままです。詳細は
> [ステータスと既知の制約](#ステータスと既知の制約) を参照してください。

## 主な特徴

- **Open JTalk をネイティブ連携** — MeCab・NJD・JPCommon を `build.rs` + `cxx-build` で
  直接コンパイルし、`pyopenjtalk` 相当のフロントエンド（`run_frontend`）を提供。
- **PyTorch チェックポイントを直接ロード** — `burn-store` の `PytorchStore` で state_dict を
  読み込み、`key_remap` で Burn モジュールのパスへ変換。
- **VITS 推論パイプライン** — text encoder / stochastic duration predictor / flow /
  HiFi-GAN decoder を移植し、音素 ID 列から音声波形を生成。
- **数値一致を重視** — Python（ESPnet2）側で生成した fixture と相対誤差で比較するテストを
  `Wgpu` / `Flex` の両バックエンドで実行。

## 動作要件

- Rust（edition 2021、最新の安定版を推奨）
- C/C++ コンパイラ
  - Windows: MSVC（Visual Studio Build Tools）
  - Linux/macOS: GCC / Clang
- Open JTalk の辞書
  - 環境変数 `OPEN_JTALK_DICT_DIR` で指定するか、実行ファイル隣接ディレクトリに配置。
  - 見つからない場合は初回実行時に
    [r9y9/open_jtalk](https://github.com/r9y9/open_jtalk) のリリースから自動ダウンロードします。

CMake は不要です。Open JTalk は Git サブモジュール `lib/open_jtalk` として同梱されます
（クローン時は `git submodule update --init --recursive`）。

## ビルド

```bash
# デバッグビルド
cargo build

# 最適化ビルド
cargo build --release
```

## 使い方

### テキスト → 音素トークン → 音素 ID

```rust
use espnet_burn::text::phoneme_tokenizer::pyopenjtalk_g2p_prosody;
use espnet_burn::text::token_id_converter::TokenIDConverter;

fn main() -> Result<(), espnet_burn::error::Error> {
    // HTS ラベルから音素＋韻律記号を生成
    let tokens = pyopenjtalk_g2p_prosody("こんにちは", true)?;
    assert_eq!(
        tokens,
        ["^", "k", "o", "[", "N", "n", "i", "ch", "i", "w", "a", "$"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    );

    // トークン → 音素 ID
    let converter = TokenIDConverter::from_file("token_list.txt", None)?;
    let ids = converter.tokens2ids(&tokens);
    println!("{ids:?}");
    Ok(())
}
```

`run_frontend` はクレートルートからも利用できます（HTS フルコンテキストラベルを返します）。

```rust
let labels = espnet_burn::run_frontend("おはよう", false)?;
```

### テキスト → 音声波形（VITS 推論）

`Text2Speech` は任意の Burn バックエンド `B` に対してジェネリックです。
利用側で `burn` のバックエンド機能（`wgpu` / `flex` など）を有効化してください。

```rust
use std::path::Path;
use burn::backend::Wgpu;
use burn::prelude::*;
use espnet_burn::tts::inference::Text2Speech;
use espnet_burn::tts::espnet_tts::model::{
    gan::vits::VITSSynthesisOptions, TTSSynthesisOptions,
};

let device = Default::default();
let tts = Text2Speech::<Wgpu>::new(
    Some(Path::new("tmp/model/config.yaml")),   // ESPnet2 config.yaml
    Some(Path::new("tmp/model/100epoch.pth")),  // チェックポイント
    &device,
    None, // seed（既定 777）
    TTSSynthesisOptions::VITS(VITSSynthesisOptions {
        alpha: 1.0,
        noise_scale: 0.667,
        noise_scale_dur: 0.8,
    }),
)?;

let ids: Vec<i64> = /* 音素 ID 列 */;
let text = Tensor::<Wgpu, 1, Int>::from_data(ids, &device);
let output = tts.forward(text, None)?;
let wav = output.wav; // Tensor<Wgpu, 1>
```

## アーキテクチャ

```
日本語テキスト
  ↓ run_frontend()            （Open JTalk: text2mecab → MeCab → NJD → JPCommon → HTS ラベル）
HTS フルコンテキストラベル
  ↓ pyopenjtalk_g2p_prosody() （ラベルから音素・韻律記号を抽出）
音素トークン列
  ↓ TokenIDConverter         （トークン → 音素 ID）
音素 ID 列
  ↓ ESPnetTTSModel::inference（VITS: text_encoder → duration → flow → decoder）
音声波形
```

主なモジュール:

| パス | 役割 |
| --- | --- |
| `openjtalk/` | Open JTalk 連携（CXX FFI、辞書解決） |
| `text/` | 音素化（`pyopenjtalk_g2p_prosody`）とトークン ID 変換 |
| `modules/` | HiFi-GAN / WaveNet / Residual Coupling |
| `tts/` | TTS パイプライン（config / task / inference / VITS 生成器） |
| `error/` | 共通エラー型 `Error` |

リポジトリの主要な配置:

| パス | 内容 |
| --- | --- |
| `src/` | ライブラリ本体（`openjtalk/` / `text/` / `modules/` / `tts/` / `error/`） |
| `lib/` | CXX ブリッジ（`openjtalk_bridge.{h,cpp}`）と Open JTalk サブモジュール `open_jtalk/` |
| `build.rs` | Open JTalk の C/C++ ビルド設定 |
| `tests/` | 数値検証テストと fixture 生成スクリプト（`tests/scripts/`） |

詳細な構成・移植方針・規約は [`AGENTS.md`](AGENTS.md) を参照してください。

## テスト

`tests/fixtures/`（`.gitignore` 対象）は Python 側で生成する必要があります。

```bash
# 単一モジュールの weight / input / output を生成
python tests/scripts/gen_fixture.py \
    --module-name <dir> \
    --weight-key-prefix "<prefix.>" \
    --input-shape "1,10,256"

# Text2Speech 全体の入出力を生成
python tests/scripts/tts2speech.py

cargo test
```

`tts` 系テストは `tmp/model/config.yaml` と `tmp/model/100epoch.pth` を参照します。
fixture が不要なテスト（例: `src/text/mod.rs` の `test_text2ids`）はそのまま実行できます。

## 開発

```bash
cargo fmt --check     # 整形チェック
cargo clippy -- -D warnings
```

開発者向けの詳細は [`AGENTS.md`](AGENTS.md) を参照してください。

## ステータスと既知の制約

本ライブラリは **COEIROINK 向けの推論**を目的として移植しています。そのため、
COEIROINK のユースケースで使用しない機能は意図的に未実装です。

推論（VITS・teacher forcing なし）の経路を中心に移植が進んでいます。以下は未対応です。

- `use_teacher_forcing=true`（`posterior_encoder` 未移植）
- 話者・言語 conditioning（`spembs` / `sids` / `lids`）
- `normalize` / `pitch_extract` / `energy_extract` 系（config で指定すると `panic!`）
- `src/task/`（`lib.rs` で無効化。未統合）

## 出典・ライセンス

本リポジトリは [Apache License 2.0](LICENSE) の下で公開されています。

移植元は [ESPnet](https://github.com/espnet/espnet)（Apache License 2.0）です。
各ファイル冒頭の `//! Ported from ...` に個別の出典を記載しています。
Open JTalk / MeCab / NAIST-jdic（BSD）や MIT 由来コードを含む第三者コンポーネントの
著作権表示とライセンス全文は [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md)、
帰属表示は [`NOTICE`](NOTICE) を参照してください。
