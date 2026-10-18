# espnet-burn 開発ガイド（AGENTS.md）

このドキュメントは、AI コーディングエージェントおよび開発者向けに、リポジトリの構成・
ビルド方法・移植方針・規約をまとめたものです。利用者向けの説明は `README.md` を参照してください。

## プロジェクト概要

**espnet-burn** は、ESPnet2 の音声合成（TTS）モデル、特に **VITS** の推論処理を
[Burn](https://burn.dev) フレームワークへ移植した Rust ライブラリです。

- 日本語テキストの解析・音素化は **Open JTalk** を CXX FFI 経由で直接呼び出して行う。
- 学習済みの PyTorch チェックポイント（`.pth`）を `burn-store` で直接ロードする。
- 入力テキスト（音素 ID 列）から音声波形を生成するまでを Rust のみで完結させることを目指す。
- **移植の主目的は COEIROINK の推論用途**であり、COEIROINK が使用しない機能は意図的に
  未実装のまま残している（後述の「既知の制約・未実装」を参照）。

## 前提環境

- Rust（edition 2021、最新の安定版を推奨）
- C/C++ コンパイラ
  - Windows: MSVC（Visual Studio Build Tools）
  - Linux/macOS: GCC / Clang
- CMake は不要（Open JTalk の C/C++ ソースは `build.rs` + `cxx-build` で直接コンパイルする）

### 主要な依存（`Cargo.toml`）

- `burn` 0.21.0（`std` のみ）、`burn-store` 0.21.0（`pytorch` / `std`）
- `cxx` / `cxx-build` 1.0.199（揃えること）
- `regex` / `serde` / `yaml_serde`（`config.yaml` のパース）/ `thiserror`
- 辞書の自動ダウンロード用: `ureq` / `tar` / `flate2`
- dev-dependencies: `burn`（`wgpu` / `flex`）— テストは両バックエンドで実行する

ライブラリ自体はバックエンド機能を有効化していないため、VITS 推論を利用する側で
`burn` のバックエンド（`wgpu` / `flex` など）を自前の依存として有効化する必要がある。

## ビルド・テスト・Lint コマンド

```bash
# デバッグビルド
cargo build

# 最適化ビルド
cargo build --release

# テスト（fixture が必要なものは後述の準備を参照）
cargo test
cargo test -- --nocapture
cargo test test_name -- --exact

# 整形 / Lint
cargo fmt --check
cargo fmt
cargo clippy -- -D warnings
```

### テスト用 fixture の準備

`tests/fixtures/` は `.gitignore` 対象でリポジトリに含まれない。数値検証テストを実行する前に、
Python（ESPnet2 + PyTorch）側で fixture を生成する。

```bash
# 単一モジュールの weight/input/output を生成（テンプレート）
python tests/gen_fixture.py

# Text2Speech 全体の入出力を生成
python tests/scripts/tts2speech.py
```

`tts` 系テストは `tmp/model/config.yaml` と `tmp/model/100epoch.pth` を参照する。
（`gen_fixture.sh` は `gen_fixture.py` の一括生成用ラッパー。）

## アーキテクチャ

### モジュール構成（`src/`）

- **`lib.rs`** — クレートルート。`run_frontend` を再公開。`task` は現在コメントアウト（無効）。
- **`error/mod.rs`** — `thiserror` ベースの共通エラー型 `Error`（`Argument` / `Io` / `Parse` /
  `Regex` / `Tokenizer` / `Request` / `Other`）。
- **`openjtalk/`** — Open JTalk 連携
  - `mod.rs` — CXX ブリッジ定義と `run_frontend(text, use_unvoiced_vowel)`。初回辞書ロードは
    `OnceLock` でキャッシュ。
  - `dict.rs` — 辞書ディレクトリの解決。`OPEN_JTALK_DICT_DIR` → 実行ファイル隣接 → 無ければ
    リリースから自動ダウンロード・展開。
  - `IMPLEMENTATION_GUIDE.md` — Open JTalk フロントエンド実装メモ。
- **`text/`** — テキスト処理
  - `phoneme_tokenizer.rs` — `pyopenjtalk_g2p_prosody()`。HTS ラベルから音素＋韻律記号を生成。
  - `token_id_converter.rs` — `TokenIDConverter`。トークン列と ID の相互変換。
- **`modules/`** — 再利用可能なニューラルネット部品（ESPnet2 からの移植）
  - `hifi_gan/generator.rs` — HiFi-GAN 生成器
  - `wavenet.rs` — グローバル条件付け付き WaveNet
  - `residual_coupling.rs` — VITS の Residual Affine Coupling（flow）
- **`tts/`** — TTS パイプライン
  - `config.rs` — ESPnet2 の `config.yaml` に対応する設定構造体
  - `task.rs` — `TTSTask`。config/checkpoint からモデルを構築・重みロード
  - `inference.rs` — `Text2Speech`（推論エントリ）と `SynthesisOutput`
  - `utils/duration_calculator.rs` — attention からの duration / focus rate 算出
  - `espnet_tts/` — ESPnet TTS モデル本体
    - `mod.rs` — `ESPnetTTSModel`（feats/pitch/energy/normalize/tts の合成）
    - `feats_extract/linear_spectrogram.rs` — 線形スペクトログラム抽出
    - `model/utils.rs` — 活性化関数ヘルパ（`forward_activation`）
    - `model/gan/vits/generator/` — VITS 生成器
      - `text_encoder/` — Conformer ベースのテキストエンコーダ
        （`encoder/layer/{attention, conformer_conv, multi_layered_conv, positional_encoding}`）
      - `duration_predictor/` — Stochastic Duration Predictor
        （`log_flow.rs`、`flow_step/{affine, flip, conv::dds}`）
      - `flow.rs` — 残差結合フロー / `decoder.rs` — HiFi-GAN デコーダ
      - `posterior_encoder.rs` — ダミーのみ（`mod` 未宣言で未使用）
    - `mod gan/vits.rs` — `VITS` / `VITSConfig` / `VITSSynthesisOptions` / `VITSynthesisOutput`
    - `normalize` / `pitch_extract` / `energy_extract` — 現状は `None` プレースホルダ
- **`task/`** — `src/task/{mod.rs,tts.rs}` は存在するが、`lib.rs` では無効化されている（未統合）。

### データフロー

```
日本語テキスト
  ↓ run_frontend()            （Open JTalk: text2mecab → MeCab → NJD → JPCommon → HTS ラベル）
HTS フルコンテキストラベル
  ↓ pyopenjtalk_g2p_prosody() （ラベルから音素・韻律記号を抽出）
音素トークン列（^ $ ? _ [ ] # など）
  ↓ TokenIDConverter         （トークン → 音素 ID）
音素 ID 列
  ↓ ESPnetTTSModel::inference（VITS: text_encoder → duration → flow → decoder）
音声波形
```

### 韻律記号

- `^` 文頭（sil） / `$` 文末（平叙） / `?` 疑問形
- `_` ポーズ（pau）
- `[` アクセント句の上昇 / `]` 下降
- `#` アクセント句境界
- 母音: `A/I/U/E/O`（有声）、`a/i/u/e/o`（無声、`drop_unvoiced_vowel=true` 時）

### FFI・ビルド

- `build.rs` が `cxx_build::bridge("src/openjtalk/mod.rs")` でブリッジを生成し、
  `lib/openjtalk_bridge.cpp`（ヘッダは `lib/openjtalk_bridge.h`）と Open JTalk の C/C++ ソース
  （MeCab 本体・NJD・JPCommon 等）をコンパイルする。
- C++/CXX のバージョンは `cxx` / `cxx-build` で揃える（`= 1.0.199`）。
- Open JTalk 本体はサブモジュール `lib/open_jtalk`（r9y9/open_jtalk）。クローン時は
  `git submodule update --init --recursive` が必要。

## 移植方針・テスト

ESPnet2（PyTorch）からの移植は、**fixture による数値一致検証**を基本とする。

1. Python 側で対象モジュールの `weight.pth` / `input.pth` / `output.pth` を生成
   （`tests/scripts/gen_fixture.py`）。
2. Rust 側は `burn_store::PytorchStore` で重みをロードし、`PytorchReader` で入出力を読む。
3. Burn の forward 結果と期待出力を**相対誤差**で比較する（`REL_TOL`、ゼロ近傍は絶対誤差下限
   `ABS_EPS` を使用）。テストは `Wgpu` と `Flex` の両バックエンドで実行する。

fixture を必要としないテストもある（例: `text/mod.rs` の `test_text2ids`。Open JTalk と
`TokenIDConverter` だけで完結する）。

### 重みキーのマッピング（`key_remap`）

Python の state_dict と Burn モジュールのパスは一致しないため、各モジュールが `key_remap` を
実装して store のキーを変換する。

- Weight Norm 保存（`*.weight_g` / `*.weight_v`）は `weight = g * v / ||v||` に融合する。
- `duration_predictor` の DDS はキー変換の副作用（`map_indices_contiguous`）があるため、
  **呼び出し順序に注意**（`duration_predictor` を先に処理する。VITS 生成器のコメント参照）。

## コーディング規約

- モジュールは `mod.rs` で宣言し、サブモジュールを整理する。
- 移植元の Python コードが分かるように、ファイル冒頭に `//! Ported from ...` 形式で出典を書く。
- 関数名は省略せず説明的にする（`g2p` など音声分野の慣用略語は可）。
- 失敗し得る公開 API は `Result<T, Error>` を返し、`Error` に説明的な内容を入れる。
- コメントは移植上の非自明な差分・制約の説明に限定する。

## 既知の制約・未実装

移植の主目的は **COEIROINK の推論用途**である。COEIROINK が使用しない機能は意図的に
未実装のまま残しているため、以下の項目は「不具合」ではなく「対象外」である点に注意する。

- `src/task` は `lib.rs` で無効化（`// pub mod task;`）。`src/task/tts.rs` は未統合。
- `posterior_encoder` は未移植（`posterior_encoder.rs` はダミーのみで `mod` も未宣言）。
  `use_teacher_forcing=true` は未対応。
- 話者・言語 conditioning（`spembs` / `sids` / `lids`）は未対応。
- `normalize` / `pitch_extract` / `energy_extract` / `pitch_normalize` / `energy_normalize`
  は現状 `None` プレースホルダ（config で指定されると `panic!`）。
- `phoneme_tokenizer.rs` は正規表現を呼び出しごとにコンパイルしている（`once_cell` 等での
  キャッシュが今後の最適化候補）。
- Open JTalk 辞書は未検出時にネットワークから自動ダウンロードする。

## 参考リンク

- [Burn ドキュメント](https://burn.dev)
- [CXX](https://github.com/dtolnay/cxx)
- [Open JTalk (r9y9 fork)](https://github.com/r9y9/open_jtalk)
- [ESPnet](https://github.com/espnet/espnet)
- ラベル形式: HTS Full Context Label Format
