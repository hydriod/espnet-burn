use std::{collections::HashMap, path::Path};

use burn::prelude::*;
use burn::tensor::TensorKind;

use crate::error::Error;
use crate::tts::espnet_tts::model::{TTSSynthesisOptions, TTSSynthesisOutput};
use crate::tts::espnet_tts::ESPnetTTSModel;
use crate::tts::task::TTSTask;
use crate::tts::utils::DurationCalculator;

pub struct Text2Speech<B: Backend> {
    model: ESPnetTTSModel<B>,
    duration_calculator: DurationCalculator<B>,
    seed: u64,
    decode_conf: TTSSynthesisOptions,
}

impl<B: Backend> Text2Speech<B> {
    pub fn new(
        train_config: Option<&Path>,
        model_config: Option<&Path>,
        device: &B::Device,
        seed: Option<u64>,
        options: TTSSynthesisOptions,
    ) -> Result<Self, Error> {
        let (model, args) = TTSTask::build_model_from_file(train_config, model_config, device)?;
        let duration_calculator = DurationCalculator::<B>::new();
        let seed = seed.unwrap_or(777);
        let decode_conf = options;

        Ok(Self {
            model,
            duration_calculator,
            seed,
            decode_conf,
        })
    }

    pub fn forward(
        &self,
        text: Tensor<B, 1, Int>,
        decode_conf: TTSSynthesisOptions,
    ) -> Result<SynthesisOutput<B>, Error> {
        let result = self.model.inference(
            text,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            decode_conf,
            false,
        )?;

        let wav = match &result {
            TTSSynthesisOutput::VITS(vit_output) => vit_output.generated.clone(),
            _ => {
                return Err(Error::Argument(
                    "Only VITS model is supported for inference.".to_string(),
                ))
            }
        };

        let attention_weight = match &result {
            TTSSynthesisOutput::VITS(vit_output) => Some(vit_output.attention_weight.clone()),
            _ => None,
        };

        let duration_result = match &result {
            TTSSynthesisOutput::VITS(vit_output) => Some(
                self.duration_calculator
                    .forward(vit_output.attention_weight.clone()),
            ),
            _ => None,
        };
        let (duration, focus_rate) = match duration_result {
            Some(duration_result) => (
                Some(duration_result.duration),
                Some(duration_result.focus_rate),
            ),
            None => (None, None),
        };

        Ok(SynthesisOutput {
            wav,
            attention_weight,
            duration,
            focus_rate,
        })
    }

    /// Alias `forward` method.
    pub fn __call__(
        &self,
        text: Tensor<B, 1, Int>,
        decode_conf: TTSSynthesisOptions,
    ) -> Result<SynthesisOutput<B>, Error> {
        self.forward(text, decode_conf)
    }

    /// Converts the string to a token ID. <br>
    /// However, the conversion is performed by COEIROINK, so nothing is done here.
    fn preprocess_fn(&self, uid: &str, data: &mut HashMap<&str, Vec<f64>>) {}
}

#[derive(Clone, Debug)]
pub struct SynthesisOutput<B: Backend> {
    pub wav: Tensor<B, 1>,
    pub attention_weight: Option<Tensor<B, 2>>,
    pub duration: Option<Tensor<B, 1>>,
    pub focus_rate: Option<Tensor<B, 1>>,
}

#[cfg(test)]
mod tests {
    use crate::tts::espnet_tts::model::gan::VITSSynthesisOptions;

    use super::*;
    use burn::backend::flex::Flex;
    use burn::backend::wgpu::Wgpu;
    use burn_store::{pytorch::PytorchReader, ModuleSnapshot, PytorchStore};

    const REL_TOL: f32 = 0.05;
    const FIXTURE_DIR: &str = "tests/fixtures/tts/";
    const MODEL_PATH: &str = "tmp/model";
    type ForwardOutput<B> = SynthesisOutput<B>;
    type TestModule<B> = Text2Speech<B>;

    fn init_module<B: Backend>(device: &B::Device) -> TestModule<B> {
        Text2Speech::new(
            Some(Path::new(MODEL_PATH).join("config.yaml").as_path()),
            Some(Path::new(MODEL_PATH).join("100epoch.pth").as_path()),
            device,
            None,
            TTSSynthesisOptions::VITS(VITSSynthesisOptions {
                alpha: 1.0,
                noise_scale: 0.0,
                noise_scale_dur: 0.0,
            }),
        )
        .unwrap_or_else(|e| panic!("モデル初期化に失敗\n{e}"))
    }

    fn forward<B: Backend>(
        module: &TestModule<B>,
        input_reader: &PytorchReader,
        device: &B::Device,
    ) -> ForwardOutput<B> {
        // forward内で生成される乱数の影響を消すためnoise_scaleを0.0に設定
        module
            .forward(
                get_tensor(input_reader, "text", device).int(),
                TTSSynthesisOptions::VITS(
                VITSSynthesisOptions {
                    alpha: 1.0,
                    noise_scale: 0.0,
                    noise_scale_dur: 0.0,
                }),
            )
            .unwrap_or_else(|e| panic!("forwardに失敗。\n{e}"))
    }

    /// 全要素が相対誤差 `tol` 以内かをチェックする。
    /// 期待値が0に近い要素は絶対誤差(ABS_EPS)を分母の下限にして判定する。
    fn assert_allclose<B: Backend>(actual: ForwardOutput<B>, expected: &PytorchReader, tol: f32) {
        // ↓ 出力の次元数(D)もPython側 run_forward() の出力shapeに合わせて置き換える
        let actual_tensor = actual.wav;
        let expected_tensor = get_tensor(expected, "wav", &Default::default());
        assert_close(actual_tensor, expected_tensor, tol);

        let actual_tensor = actual.attention_weight.unwrap();
        let expected_tensor = get_tensor(expected, "att_w", &Default::default());
        assert_close(actual_tensor, expected_tensor, tol);

        let actual_tensor = actual.duration.unwrap();
        let expected_tensor = get_tensor(expected, "duration", &Default::default());
        assert_close(actual_tensor, expected_tensor, tol);

        let actual_tensor = actual.focus_rate.unwrap();
        let expected_tensor = get_tensor(expected, "focus_rate", &Default::default());
        assert_close(actual_tensor, expected_tensor, tol);
    }

    // --- 以下変更不要 ---
    #[test]
    fn test_wgpu() {
        numerical_check::<Wgpu>();
    }

    #[test]
    fn test_flex() {
        numerical_check::<Flex>();
    }

    fn numerical_check<TestBackend: Backend>() {
        let device = Default::default();
        let module = init_module::<TestBackend>(&device);

        // --- 3. Python側で生成した入力をロードする ---
        let input_path = format!("{FIXTURE_DIR}/input.pth");
        let input_reader = PytorchReader::new(&input_path)
            .unwrap_or_else(|e| panic!("input.pth の読み込みに失敗しました:\n{e}"));

        // --- 4. Burn側 forward を実行する ---
        let actual_output = forward(&module, &input_reader, &device);

        // --- 5. Python側の期待出力をロードする ---
        let output_path = format!("{FIXTURE_DIR}/output.pth");
        let output_reader = PytorchReader::new(&output_path)
            .unwrap_or_else(|e| panic!("output.pth の読み込みに失敗しました:\n{e}"));

        // --- 6. 比較 (相対誤差 REL_TOL 以内) ---
        assert_allclose(actual_output, &output_reader, REL_TOL);
    }

    fn assert_close<B: Backend, const D: usize>(
        actual: Tensor<B, D>,
        expected: Tensor<B, D>,
        tol: f32,
    ) {
        assert_eq!(
            actual.dims(),
            expected.dims(),
            "次元数が一致しません: actual={}, expected={}",
            actual.dims().map(|d| d.to_string()).join(", "),
            expected.dims().map(|d| d.to_string()).join(", ")
        );

        const ABS_EPS: f32 = 0.01;

        let rel_err =
            (actual.clone() - expected.clone()).abs() / expected.clone().abs().clamp_min(ABS_EPS);
        let max_rel_err = to_vec(rel_err.clone().max())[0];
        let max_rel_err_idx = to_vec(rel_err.reshape([-1]).argmax(0).float())[0] as usize;

        assert!(
            max_rel_err <= tol,
            "相対誤差が許容値を超えています: max_rel_err={:.6} (tol={:.6}) at index={}, actual={}, expected={}",
            max_rel_err,
            tol,
            max_rel_err_idx,
            to_vec(actual)[max_rel_err_idx],
            to_vec(expected)[max_rel_err_idx],
        );
    }

    /// N次元テンソルをVec<f32>に変換するヘルパー。
    fn to_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap_or_else(|e| panic!("テンソルデータの変換に失敗しました:\n{e}"))
    }

    fn get_tensor<B: Backend, const D: usize>(
        reader: &PytorchReader,
        key: &str,
        device: &B::Device,
    ) -> Tensor<B, D> {
        let data = reader
            .get(key)
            .unwrap_or_else(|| panic!("{} が見つかりません", key))
            .to_data()
            .unwrap_or_else(|e| panic!("テンソルデータの取得に失敗しました:\n{e}"));
        Tensor::from_data(data, device)
    }
}
