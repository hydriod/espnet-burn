use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig1d, GLU};
use burn::prelude::*;
use burn_store::PytorchStore;

use crate::tts::espnet_tts::model::utils::forward_activation;

pub struct ConformerConvConfig {
    channels: usize,
    kernel_size: usize,
    bias: bool,
    activation: String,
}
impl ConformerConvConfig {
    pub fn new(channels: usize, kernel_size: usize, bias: bool, activation: String) -> Self {
        Self {
            channels,
            kernel_size,
            bias,
            activation,
        }
    }
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConformerConv<B> {
        let pointwise_conv1 = Conv1dConfig::new(self.channels, 2 * self.channels, self.kernel_size)
            .with_bias(self.bias)
            .with_padding(PaddingConfig1d::Valid)
            .init(device);
        let glu = GLU::new(1);
        let depthwise_conv = Conv1dConfig::new(self.channels, self.channels, self.kernel_size)
            .with_bias(self.bias)
            .with_padding(PaddingConfig1d::Same)
            .init(device);
        let norm = BatchNormConfig::new(self.channels).init(device);
        let activation = self.activation.clone();
        let pointwise_conv2 = Conv1dConfig::new(self.channels, self.channels, 1)
            .with_bias(self.bias)
            .with_padding(PaddingConfig1d::Valid)
            .init(device);

        ConformerConv {
            pointwise_conv1,
            glu,
            depthwise_conv,
            norm,
            activation,
            pointwise_conv2,
        }
    }
}

#[derive(Module, Debug)]
pub struct ConformerConv<B: Backend> {
    pointwise_conv1: Conv1d<B>,
    glu: GLU,
    depthwise_conv: Conv1d<B>,
    norm: BatchNorm<B>,
    activation: String,
    pointwise_conv2: Conv1d<B>,
}
impl<B: Backend> ConformerConv<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        // exchange the temporal dimension and the feature dimension
        let x = x.swap_dims(1, 2);

        // GLU mechanism
        let x = self.pointwise_conv1.forward(x); // (batch, 2*channel, dim)
        let x = self.glu.forward(x); // (batch, channel, dim)

        // 1D Depthwise Conv
        let x = Self::mask_padded_frames(x, mask);
        let x = self.depthwise_conv.forward(x);
        let x = self.norm.forward(x);
        let x = forward_activation(x, &self.activation);

        let x = self.pointwise_conv2.forward(x);

        return x.swap_dims(1, 2);
    }

    fn mask_padded_frames(x: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        if mask.dims()[1] != 1 && x.dims()[2] != mask.dims()[2] {
            x.mask_fill(mask.equal_elem(0), 0)
        } else {
            x
        }
    }

    pub fn key_remap(&self, store: PytorchStore, parents: &Vec<&str>) -> PytorchStore {
        store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::WgpuDevice;
    use burn::backend::Wgpu;
    use burn_store::{pytorch::PytorchReader, ModuleSnapshot, PytorchStore};

    const REL_TOL: f32 = 0.05;
    type TestDevice = WgpuDevice;
    type TestBackend = Wgpu;

    const FIXTURE_DIR: &str =
        "tests/fixtures/vits/generator/text_encoder/encoder/layer/conformer_conv";
    type ForwardOutput = Tensor<TestBackend, 3>;
    type TestModule = ConformerConv<TestBackend>;

    fn init_module(device: &TestDevice) -> TestModule {
        ConformerConvConfig::new(192, 3, true, "swish".to_string()).init(device)
    }

    fn key_remap(store: PytorchStore, module: &TestModule) -> PytorchStore {
        let store = module.key_remap(store, &vec![]);
        store
    }
    fn forward(
        module: &TestModule,
        input_reader: &PytorchReader,
        device: &TestDevice,
    ) -> ForwardOutput {
        module.forward(
            get_tensor(input_reader, "x", device),
            get_tensor(input_reader, "mask", device),
        )
    }

    /// 全要素が相対誤差 `tol` 以内かをチェックする。
    /// 期待値が0に近い要素は絶対誤差(ABS_EPS)を分母の下限にして判定する。
    fn assert_allclose(actual: ForwardOutput, expected: &PytorchReader, tol: f32) {
        // ↓ 出力の次元数(D)もPython側 run_forward() の出力shapeに合わせて置き換える
        let actual_output_0 = actual;
        let expected_output_0 = get_tensor(expected, "output_0", &Default::default());

        assert_close(actual_output_0, expected_output_0, tol);
    }

    // --- 以下変更不要 ---

    #[test]
    fn test_forward() {
        let device = Default::default();
        let mut module = init_module(&device);

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

    fn assert_close<const D: usize>(
        actual: Tensor<TestBackend, D>,
        expected: Tensor<TestBackend, D>,
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
    fn to_vec<const D: usize>(t: Tensor<TestBackend, D>) -> Vec<f32> {
        t.into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap_or_else(|e| panic!("テンソルデータの変換に失敗しました:\n{e}"))
    }

    fn get_tensor<const D: usize>(
        reader: &PytorchReader,
        key: &str,
        device: &TestDevice,
    ) -> Tensor<TestBackend, D> {
        let data = reader
            .get(key)
            .unwrap_or_else(|| panic!("{} が見つかりません", key))
            .to_data()
            .unwrap_or_else(|e| panic!("テンソルデータの取得に失敗しました:\n{e}"));
        Tensor::from_data(data, device)
    }
}
