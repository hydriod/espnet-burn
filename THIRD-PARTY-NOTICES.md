# 第三者ライセンス（Third-Party Notices）

このリポジトリ（espnet-burn）は、以下の第三者ソフトウェアを利用・翻訳・改変（移植）しています。
各コンポーネントの著作権表示とライセンス条件を以下に記載します。

本リポジトリ自身のライセンスは `LICENSE`（Apache License 2.0）を参照してください。

---

## 1. ESPnet — Apache License 2.0

本リポジトリの Rust コードの多くは、ESPnet2
(<https://github.com/espnet/espnet>) の Python 実装を Burn へ移植したものです。

- Copyright 2017 Shinji Watanabe
- 移植対象モジュールの著作権: Copyright 2021 Tomoki Hayashi
- ライセンス: Apache License 2.0（全文は [`LICENSE`](LICENSE)）

移植元の各ファイルは、リポジトリ内でもファイル冒頭に `//! Ported from ...` として出典を
明記しています。

---

## 2. ParallelWaveGAN — MIT License

`src/modules/hifi_gan/generator.rs` および `src/modules/wavenet.rs` は、ESPnet2 経由で
ParallelWaveGAN (<https://github.com/kan-bayashi/ParallelWaveGAN>) のコードを基にしています。

```
MIT License

Copyright (c) 2020 Tomoki Hayashi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 3. VITS — MIT License

本リポジトリの VITS 関連モジュール（`src/tts/espnet_tts/model/gan/vits/` など）は、
VITS (<https://github.com/jaywalnut310/vits>) を起源とする実装を移植したものです。

```
MIT License

Copyright (c) 2021 Jaehyeon Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 4. Open JTalk — Modified BSD (3-clause) License

`lib/open_jtalk`（Git サブモジュール、r9y9/open_jtalk フォーク）および
`lib/openjtalk_bridge.{h,cpp}` 経由でビルド・リンクされるネイティブコード。

出典: <https://open-jtalk.sourceforge.net/>

```
The Japanese TTS System "Open JTalk"
developed by HTS Working Group
http://open-jtalk.sourceforge.net/

Copyright (c) 2008-2018  Nagoya Institute of Technology
                         Department of Computer Science

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of the HTS working group nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

---

## 5. MeCab — BSD (3-clause) License

Open JTalk に同梱され、本リポジトリのビルドに含まれる形態素解析エンジン
（`lib/open_jtalk/src/mecab`）。

出典: <https://taku910.github.io/mecab/>

```
Copyright (c) 2001-2008, Taku Kudo
Copyright (c) 2004-2008, Nippon Telegraph and Telephone Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name of the Nippon Telegraph and Telegraph Corporation nor the
   names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

---

## 6. NAIST-jdic — BSD (3-clause) License

`OPEN_JTALK_DICT_DIR` が未設定かつ実行ファイル隣接に辞書が無い場合に、実行時に
自動ダウンロードされる日本語辞書。

出典: <https://sourceforge.jp/projects/naist-jdic/>

```
Copyright (c) 2009, Nara Institute of Science and Technology, Japan.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
Neither the name of the Nara Institute of Science and Technology (NAIST)
nor the names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

---

## 7. Rust クレート依存

`Cargo.toml` / `Cargo.lock` で参照する Rust クレート。直接依存は下表のとおりです
（いずれもパーミッシブライセンス）。

| クレート | バージョン | ライセンス |
| --- | --- | --- |
| burn | 0.21.0 | MIT OR Apache-2.0 |
| burn-store | 0.21.0 | MIT OR Apache-2.0 |
| regex | 1.13.1 | MIT OR Apache-2.0 |
| cxx | 1.0.199 | MIT OR Apache-2.0 |
| cxx-build | 1.0.199 | MIT OR Apache-2.0 |
| tar | 0.4.46 | MIT OR Apache-2.0 |
| flate2 | 1.1.9 | MIT OR Apache-2.0 |
| ureq | 3.4.0 | MIT OR Apache-2.0 |
| serde | 1.0.229 | MIT OR Apache-2.0 |
| yaml_serde | 0.10.7 | MIT OR Apache-2.0 |
| thiserror | 2.0.20 | MIT OR Apache-2.0 |

> 注: 上表は直接依存のみです。推移的依存も含めた正確な SPDX 識別子は、次のコマンドで
> 機械的に確認してください。
>
> ```bash
> cargo install cargo-license
> cargo license
> ```

---

## 8. モデル重み・音声データについて

本リポジトリは学習済みモデルの重み（`.pth` など）や音声データを配布するものでは
ありません。利用するモデル・辞書・音声には、それぞれの提供元（例: COEIROINK の
利用規約）が定める別個のライセンス・規約が適用されます。
