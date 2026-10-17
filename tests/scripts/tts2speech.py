from espnet2.bin.tts_inference import Text2Speech
import torch
import os


if __name__ == "__main__":
    tts = Text2Speech(
        "tmp/model/config.yaml",
        "tmp/model/100epoch.pth",
        speed_control_alpha=1.0,
        noise_scale=0.0,
        noise_scale_dur=0.0
    )
    text = torch.randint(0, 47, (8,))
    output = tts.__call__(text)

    # テンソルを連続したメモリに変換する
    output = {
        k: (v.contiguous() if torch.is_tensor(v) else v)
        for k, v in output.items()
    }
    if torch.is_tensor(output.get("focus_rate")):
        output["focus_rate"] = output["focus_rate"].reshape(1)

    os.makedirs("tests/fixtures/tts", exist_ok=True)
    inputs = {"text": text}
    torch.save(inputs, "tests/fixtures/tts/input.pth")
    print(f"input.pth: {inputs}")
    torch.save(output, "tests/fixtures/tts/output.pth")
    print(f"output.pth: {output}")
