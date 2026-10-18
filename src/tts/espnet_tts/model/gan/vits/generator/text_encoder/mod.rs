mod encoder;

use burn::nn::conv::Conv1dConfig;
use burn::nn::EmbeddingConfig;
use burn::nn::{conv::Conv1d, Embedding};
use burn::prelude::*;
use burn::tensor::DataError;
use burn_store::PytorchStore;

use crate::error::Error;
use crate::tts::espnet_tts::model::gan::vits::generator::text_encoder::encoder::{
    Encoder, EncoderConfig,
};

pub struct TextEncoderConfig {
    vocabs: usize,
    attention_dim: usize,
    attention_heads: usize,
    linear_units: usize,
    blocks: usize,
    positionwise_layer_type: String,
    positionwise_conv_kernel_size: usize,
    positional_encoding_layer_type: String,
    self_attention_layer_type: String,
    activation_type: String,
    normalize_before: bool,
    use_macaron_style: bool,
    use_conformer_conv: bool,
    conformer_kernel_size: usize,
    dropout_rate: f64,
    positional_dropout_rate: f64,
    attention_dropout_rate: f64,
}
impl TextEncoderConfig {
    pub fn new(
        vocabs: impl TryInto<usize>,
        attention_dim: impl TryInto<usize>,
        attention_heads: impl TryInto<usize>,
        linear_units: impl TryInto<usize>,
        blocks: impl TryInto<usize>,
        positionwise_layer_type: String,
        positionwise_conv_kernel_size: impl TryInto<usize>,
        positional_encoding_layer_type: String,
        self_attention_layer_type: String,
        activation_type: String,
        normalize_before: bool,
        dropout_rate: f64,
        positional_dropout_rate: f64,
        attention_dropout_rate: f64,
        conformer_kernel_size: impl TryInto<usize>,
        use_macaron_style: bool,
        use_conformer_conv: bool,
    ) -> Result<Self, Error> {
        let vocabs = vocabs
            .try_into()
            .map_err(|_| Error::Argument(format!("Fail to try_into vocabs")))?;
        let attention_dim = attention_dim
            .try_into()
            .map_err(|_| Error::Argument(format!("Fail to try_into attention_dim")))?;
        let attention_heads = attention_heads
            .try_into()
            .map_err(|_| Error::Argument(format!("Fail to try_into attention_heads")))?;
        let linear_units = linear_units
            .try_into()
            .map_err(|_| Error::Argument(format!("Fail to try_into linear_units")))?;
        let blocks = blocks
            .try_into()
            .map_err(|_| Error::Argument(format!("Fail to try_into blocks")))?;
        let positionwise_conv_kernel_size =
            positionwise_conv_kernel_size.try_into().map_err(|_| {
                Error::Argument(format!("Fail to try_into positionwise_conv_kernel_size"))
            })?;
        let conformer_kernel_size = conformer_kernel_size
            .try_into()
            .map_err(|_| Error::Argument(format!("Fail to try_into conformer_kernel_size")))?;
        Ok(Self {
            vocabs,
            attention_dim,
            attention_heads,
            linear_units,
            blocks,
            positionwise_layer_type,
            positionwise_conv_kernel_size,
            positional_encoding_layer_type,
            self_attention_layer_type,
            activation_type,
            normalize_before,
            use_macaron_style,
            use_conformer_conv,
            conformer_kernel_size,
            dropout_rate,
            positional_dropout_rate,
            attention_dropout_rate,
        })
    }
    pub fn init<B: Backend>(&self, device: &B::Device) -> TextEncoder<B> {
        TextEncoder {
            emb: EmbeddingConfig::new(self.vocabs, self.attention_dim).init(device),
            encoder: EncoderConfig::new(
                "".to_string(),
                self.attention_dim,
                self.attention_heads,
                self.linear_units,
                self.blocks,
                self.dropout_rate,
                self.positional_dropout_rate,
                self.attention_dropout_rate,
                self.normalize_before,
                self.positionwise_layer_type.clone(),
                self.positionwise_conv_kernel_size,
                self.use_macaron_style,
                self.positional_encoding_layer_type.clone(),
                self.self_attention_layer_type.clone(),
                self.activation_type.clone(),
                self.use_conformer_conv,
                self.conformer_kernel_size,
            )
            .init(device),
            proj: Conv1dConfig::new(self.attention_dim, self.attention_dim * 2, 1).init(device),
            attention_dim_sqrt: f64::sqrt(self.attention_dim as f64),
        }
    }
}

#[derive(Module, Debug)]
pub struct TextEncoder<B: Backend> {
    emb: Embedding<B>,
    encoder: Encoder<B>,
    proj: Conv1d<B>,
    attention_dim_sqrt: f64,
}
impl<B: Backend> TextEncoder<B> {
    /// # Args
    /// - `x` : Input index tensor (B, T_text).
    /// - `x_lengths` : Length tensor (B,).
    pub fn forward(
        &self,
        x: Tensor<B, 2, Int>,
        x_lengths: Tensor<B, 1>,
    ) -> Result<TextEncoderOutput<B>, DataError> {
        let x = self.emb.forward(x) * self.attention_dim_sqrt;
        let x_mask = make_non_pad_mask(x_lengths, &x.device())?.unsqueeze_dim(1);
        let x = self.encoder.forward(x, x_mask.clone()).swap_dims(1, 2);
        let stats = self.proj.forward(x.clone()) * x_mask.clone();
        let mean_length = stats.dims()[1] / 2;
        let m = stats.clone().slice_dim(1, ..mean_length);
        let logs = stats.slice_dim(1, mean_length..);

        Ok(TextEncoderOutput {
            encoded: x,
            mean: m,
            scale: logs,
            mask: x_mask,
        })
    }

    pub fn key_remap(&self, store: PytorchStore, parents: &Vec<&str>) -> PytorchStore {
        let mut parents = parents.clone();
        let p = "encoder";
        parents.push(p);
        self.encoder.key_remap(store, &parents)
    }
}

pub struct TextEncoderOutput<B: Backend> {
    /// Encoded hidden representation (B, attention_dim, T_text).
    pub encoded: Tensor<B, 3>,
    /// Projected mean tensor (B, attention_dim, T_text).
    pub mean: Tensor<B, 3>,
    /// Projected scale tensor (B, attention_dim, T_text).
    pub scale: Tensor<B, 3>,
    /// Mask tensor for input tensor (B, 1, T_text).
    pub mask: Tensor<B, 3>,
}

fn make_non_pad_mask<B: Backend>(
    output_length: Tensor<B, 1>,
    device: &B::Device,
) -> Result<Tensor<B, 2>, DataError> {
    let target_dim = output_length
        .clone()
        .max()
        .int()
        .to_data()
        .to_vec::<i32>()?[0];
    let pad_mask = output_length
        .to_data()
        .to_vec::<f32>()?
        .iter()
        .map(|len| {
            let len = len.to_i32();
            Tensor::cat(
                vec![
                    Tensor::<B, 1>::ones([len], device),
                    Tensor::<B, 1>::zeros([target_dim - len], device),
                ],
                0,
            )
        })
        .collect::<Vec<Tensor<B, 1>>>();
    let pad_mask = Tensor::stack::<2>(pad_mask, 0);

    Ok(pad_mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::Flex;
    use burn::backend::wgpu::Wgpu;
    use burn_store::{pytorch::PytorchReader, ModuleSnapshot, PytorchStore};

    const REL_TOL: f32 = 0.05;
    const FIXTURE_DIR: &str = "tests/fixtures/vits/generator/text_encoder/mod";
    type ForwardOutput<B> = TextEncoderOutput<B>;
    type TestModule<B> = TextEncoder<B>;

    fn init_module<B: Backend>(device: &B::Device) -> TestModule<B> {
        TextEncoderConfig::new(
            47,
            192,
            2,
            768,
            6,
            "conv1d".to_string(),
            3,
            "rel_pos".to_string(),
            "rel_selfattn".to_string(),
            "swish".to_string(),
            true,
            0.1,
            0.0,
            0.1,
            0,
            true,
            false,
        )
        .unwrap()
        .init(device)
    }

    fn key_remap<B: Backend>(store: PytorchStore, module: &TestModule<B>) -> PytorchStore {
        let store = module.key_remap(store, &vec![]);
        store
    }

    fn forward<B: Backend>(
        module: &TestModule<B>,
        input_reader: &PytorchReader,
        device: &B::Device,
    ) -> ForwardOutput<B> {
        module
            .forward(
                get_tensor(input_reader, "x", device).int(),
                get_tensor(input_reader, "x_lengths", device),
            )
            .unwrap_or_else(|e| panic!("forwardに失敗。\n{e}"))
    }

    /// 全要素が相対誤差 `tol` 以内かをチェックする。
    /// 期待値が0に近い要素は絶対誤差(ABS_EPS)を分母の下限にして判定する。
    fn assert_allclose<B: Backend>(actual: ForwardOutput<B>, expected: &PytorchReader, tol: f32) {
        // ↓ 出力の次元数(D)もPython側 run_forward() の出力shapeに合わせて置き換える
        let actual_tensor = actual.encoded;
        let expected_tensor = get_tensor(expected, "output_0", &Default::default());
        assert_close(actual_tensor, expected_tensor, tol);

        let actual_tensor = actual.mean;
        let expected_tensor = get_tensor(expected, "output_1", &Default::default());
        assert_close(actual_tensor, expected_tensor, tol);

        let actual_tensor = actual.scale;
        let expected_tensor = get_tensor(expected, "output_2", &Default::default());
        assert_close(actual_tensor, expected_tensor, tol);

        let actual_tensor = actual.mask;
        let expected_tensor = get_tensor(expected, "output_3", &Default::default());
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
        let mut module = init_module::<TestBackend>(&device);

        // --- 2. 重みのロード
        let weight_path = format!("{FIXTURE_DIR}/weight.pth");
        let weight_store = PytorchStore::from_file(&weight_path);
        let mut weight_store = key_remap(weight_store, &module);
        module.load_from(&mut weight_store).unwrap_or_else(|e| {
            panic!("重みのロードに失敗しました。key_remapやshapeの不一致を確認してください:\n{e}")
        });

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
