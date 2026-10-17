"""
PyTorch -> Burn 移植テスト用 fixture 生成テンプレート

【使い方】
1. このファイルを `scripts/gen_fixture_{module_name}.py` としてコピーする
   (コピーせずこのまま使い回してもよい。以下は「複製して使う」想定で説明する)
2. 下の「ユーザー編集エリア」(build_module / run_forward) を対象モジュールに合わせて書き換える
3. リポジトリルートで実行する:

       python scripts/gen_fixture_template.py \\
           --module-name encoder \\
           --weight-key-prefix "encoder." \\
           --input-shape 1,10,256

   複数入力がある場合は --input-shape を複数回指定する:

       python scripts/gen_fixture_template.py \\
           --module-name duration_predictor \\
           --weight-key-prefix "duration_predictor." \\
           --input-shape 1,10,256 \\
           --input-shape 1,10

4. tests/fixtures/{module_name}/ 以下に
       weight.pth  … このモジュール分だけ抽出した重み (state_dict形式)
       input.pth   … forwardへの入力 (dict形式)
       output.pth  … PyTorch forwardの出力 (dict形式)
   が生成される

生成された3つの .pth は、Rust側で
  - weight.pth -> burn_store::PytorchStore でモジュールにロード
  - input.pth / output.pth -> burn_store::pytorch::PytorchReader でテンソルとして読み込み
して比較する想定。
"""

import argparse
import re
from pathlib import Path
import torch
import json

# ============================================================
# ユーザー編集エリア ここから
# ============================================================


def build_module(weight_key_prefix: str):
    """
    モジュールをインスタンス化する。
    コンストラクタ引数はモデル定義に合わせて埋める。

    `weight_key_prefix` は --weight-key-prefix で指定したprefixを
    importlib で動的にimportしたものが渡される。
    """
    from espnet2.tasks.tts import TTSTask
    model, args = TTSTask.build_model_from_file(model_file="tmp/model/100epoch.pth", device="cpu")
    model.eval()
    attributes = weight_key_prefix.split(".")
    for attr in attributes[:-1]:
        model = getattr(model, attr)
    return model


def run_forward(forward_func, inputs: dict):
    """
    forward を呼び出す。

    `inputs` は {"input_0": tensor, "input_1": tensor, ...} の形式で、
    --input-shape を指定した順に input_0, input_1, ... となっている。
    デフォルトでは input_0, input_1, ... の順に位置引数として渡す。
    forwardの実際のシグネチャ(キーワード引数が必要、一部だけ渡す等)に合わせて
    ここを書き換える。

    複数出力(tuple)の場合は展開して output_0, output_1, ... に割り当てる。
    """

    output = forward_func(**inputs)
    if isinstance(output, tuple):
        return {f"output_{i}": o.contiguous() for i, o in enumerate(output)}
    return {"output_0": output.contiguous()}


# ============================================================
# ユーザー編集エリア ここまで (以下は基本的に変更不要)
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorchモジュールからBurnテスト用fixtureを生成する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--module-name",
        required=True,
        help="出力フォルダ名。tests/fixtures/{module-name}/ に保存される (例: encoder)",
    )
    parser.add_argument(
        "-p",
        "--weight-key-prefix",
        required=True,
        help=(
            "全体state_dict内でこのモジュールのキーが持つprefix。"
            "末尾のドットを忘れないこと (例: 'encoder.')"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="tmp/model/100epoch.pth",
        help="全体の重みファイルパス (デフォルト: tmp/epoch1000.pth)",
    )
    parser.add_argument(
        "-i",
        "--input-shape",
        required=True,
        help=(
            "json形式で指定する入力テンソルの形状。"
            r"(例: --input-shape '{""x"": ""1,10,256""}')。dtypeはfloat32固定"
        ),
    )
    parser.add_argument(
        "--input-type",
        default="{}",
        help=(
            "json形式で指定する入力テンソルの形状。"
            r"(例: --input-shape '{""x"": ""randint(0,10)""}')。dtypeはfloat32固定"
        ),
    )
    parser.add_argument(
        "-f",
        "--forward-func",
        default="forward",
        help="推論に使う関数名",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="乱数入力生成に使うseed (デフォルト: 0)",
    )
    return parser.parse_args()


def extract_submodule_weights(full_state_dict: dict, prefix: str) -> dict:
    """全体 state_dict から prefix に一致するキーだけを取り出し、prefixを外す。"""
    extracted = {}
    for key, tensor in full_state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            extracted[new_key] = tensor
    if not extracted:
        print(
            f"prefix '{prefix}' に一致するキーが見つかりませんでした。"
            f" --weight-key-prefix を確認してください。"
            f" 全体のキー例: {list(full_state_dict.keys())[:5]}"
        )
    return extracted


def parse_shape(shape_str: str) -> tuple:
    try:
        shape = tuple(int(x) for x in shape_str.split(","))
    except ValueError as e:
        raise ValueError(
            f"--input-shape の形式が不正です: '{shape_str}' (例: 1,10,256)"
        ) from e
    if not shape:
        raise ValueError(f"--input-shape が空です: '{shape_str}'")
    return shape


def generate_inputs(input_shapes: dict, input_type: dict, seed: int) -> dict:
    """--input-shape で指定した順に input_0, input_1, ... の乱数テンソルを生成する。"""
    def gen_rand_input(shape, rand_type: str):
        if rand_type.startswith("randint"):
            m=re.match(r"randint\((\d+),(\d+)\)", rand_type)
            min, max = m.groups()
            return torch.randint(int(min), int(max), shape)
        else:
            return torch.randn(shape)
            
    def gen_recursive(shape, type):
        if isinstance(shape, str):
            shape = parse_shape(shape)
            type = type or ""
            return gen_rand_input(shape, type)
        elif isinstance(shape, dict):
            type = type or {}
            return {k: gen_recursive(v, type.get(k, None)) for k, v in shape.items()}
        elif isinstance(shape, list):
            type = type or [None]*shape.__len__()
            return tuple(gen_recursive(shape[i], type[i]) for i in range(len(shape)))
        else:
            raise ValueError(f"Invalid shape type: {type(shape)}")

    torch.manual_seed(seed)
    return gen_recursive(input_shapes, input_type)

def contiguous_recursive(x):
    if isinstance(x, torch.Tensor):
        return x.contiguous()
    elif isinstance(x, dict):
        return {k: contiguous_recursive(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(contiguous_recursive(v) for v in x)
    else:
        return x
        

def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, torch.Tensor]:
    """ネストされた辞書をフラット化する。キーは sep で連結する。"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                items.extend(flatten_dict({f"{new_key}{sep}{i}": item}, "", sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main():
    args = parse_args()
    input_type: dict = json.loads(args.input_type)
    input_shapes: dict = json.loads(args.input_shape)

    out_dir = Path("tests/fixtures") / args.module_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 全体チェックポイントをロード ---
    full_checkpoint = torch.load(args.checkpoint, map_location="cpu")
    full_state_dict = (
        full_checkpoint.get("state_dict", full_checkpoint)
        if isinstance(full_checkpoint, dict)
        else full_checkpoint
    )

    # --- 2. モジュール単位の重みを抽出して保存 ---
    module_weights = extract_submodule_weights(full_state_dict, args.weight_key_prefix)
    weight_path = out_dir / "weight.pth"
    torch.save(module_weights, weight_path)
    print(f"[OK] weight saved: {weight_path} ({len(module_weights)} tensors)")

    # --- 3. モジュールを構築し、抽出した重みが正しくロードできるか確認 ---
    module = build_module(args.weight_key_prefix)
    print(f"[OK] module built: {module.__class__.__name__}")

    # --- 4. 乱数入力を生成 ---
    inputs = generate_inputs(input_shapes, input_type, args.seed)

    # --- 5. forward実行 ---
    with torch.no_grad():
        outputs = run_forward(getattr(module, args.forward_func), inputs)
    outputs = contiguous_recursive(outputs)

    # --- 6. 入力、出力を保存 ---
    inputs = flatten_dict(inputs)
    input_path = out_dir / "input.pth"
    torch.save(inputs, input_path)
    print(f"[OK] input saved: {input_path}")
    outputs = flatten_dict(outputs)
    output_path = out_dir / "output.pth"
    torch.save(outputs, output_path)
    print(f"[OK] output saved: {output_path}")

    print("\n生成されたテンソルの形状:")
    for group_name, tensors in [("input", inputs), ("output", outputs)]:
        print(f"====== {group_name} ======")
        for k, v in tensors.items():
            print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")


if __name__ == "__main__":
    main()
