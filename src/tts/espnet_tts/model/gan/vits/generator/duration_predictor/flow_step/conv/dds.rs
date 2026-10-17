use std::marker::PhantomData;

use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, PaddingConfig1d};
use burn::prelude::*;
use burn::tensor::activation::gelu;
use burn_store::PytorchStore;

#[derive(Debug, Module, Clone)]
struct Transpose {
    dim1: usize,
    dim2: usize,
}
impl Transpose {
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        x.swap_dims(self.dim1, self.dim2)
    }
}

#[derive(Module, Debug)]
enum DDSConvBlockLayer<B: Backend> {
    Conv1d(Conv1d<B>),
    LayerNorm(LayerNorm<B>),
    Dropout(Dropout),
    Transpose(Transpose),
    GELU(PhantomData<B>),
}
impl<B: Backend> DDSConvBlockLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            DDSConvBlockLayer::Conv1d(conv) => conv.forward(x),
            DDSConvBlockLayer::LayerNorm(norm) => norm.forward(x),
            DDSConvBlockLayer::Dropout(dropout) => dropout.forward(x),
            DDSConvBlockLayer::Transpose(transpose) => transpose.forward(x),
            DDSConvBlockLayer::GELU(_) => gelu(x),
        }
    }
}
/// Dilated depth-separable conv block.

struct DDSConvBlockConfig {
    channels: usize,
    kernel_size: usize,
    dilation: usize,
    dropout_rate: f64,
    eps: f64,
}

impl DDSConvBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Vec<DDSConvBlockLayer<B>> {
        let padding = (self.kernel_size * self.dilation - self.dilation) / 2;
        let depthwise_conv = Conv1dConfig::new(self.channels, self.channels, self.kernel_size)
            .with_groups(self.channels)
            .with_dilation(self.dilation)
            .with_padding(PaddingConfig1d::Explicit(padding, padding))
            .init(device);
        let norm1 = LayerNormConfig::new(self.channels)
            .with_epsilon(self.eps)
            .init(device);
        let pointwise_conv = Conv1dConfig::new(self.channels, self.channels, 1).init(device);
        let norm2 = LayerNormConfig::new(self.channels)
            .with_epsilon(self.eps)
            .init(device);
        let dropout = DropoutConfig::new(self.dropout_rate).init();

        let mut convs = Vec::with_capacity(11);
        convs.push(DDSConvBlockLayer::Conv1d(depthwise_conv));
        convs.push(DDSConvBlockLayer::Transpose(Transpose { dim1: 1, dim2: 2 }));
        convs.push(DDSConvBlockLayer::LayerNorm(norm1));
        convs.push(DDSConvBlockLayer::Transpose(Transpose { dim1: 1, dim2: 2 }));
        convs.push(DDSConvBlockLayer::GELU(PhantomData));
        convs.push(DDSConvBlockLayer::Conv1d(pointwise_conv));
        convs.push(DDSConvBlockLayer::Transpose(Transpose { dim1: 1, dim2: 2 }));
        convs.push(DDSConvBlockLayer::LayerNorm(norm2));
        convs.push(DDSConvBlockLayer::Transpose(Transpose { dim1: 1, dim2: 2 }));
        convs.push(DDSConvBlockLayer::GELU(PhantomData));
        convs.push(DDSConvBlockLayer::Dropout(dropout));

        convs
    }
}

/// Dilated depth-separable conv module.
#[derive(Module, Debug)]
pub struct DilatedDepthSeparableConv<B: Backend> {
    convs: Vec<Vec<DDSConvBlockLayer<B>>>,
}

impl<B: Backend> DilatedDepthSeparableConv<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let mut x = match g {
            Some(g) => x + g,
            None => x,
        };
        for block in &self.convs {
            let mut y = x.clone() * x_mask.clone();
            for layer in block {
                y = layer.forward(y);
            }
            x = x + y;
        }
        x * x_mask
    }

    /// map_indices_contiguousをfalseにする
    pub fn key_remap(&self, store: PytorchStore, _: &Vec<&str>) -> PytorchStore {
        store.map_indices_contiguous(false)
    }
}

pub struct DilatedDepthSeparableConvConfig {
    channels: usize,
    kernel_size: usize,
    layers: usize,
    dropout_rate: f64,
    eps: f64,
}

impl DilatedDepthSeparableConvConfig {
    pub fn new(channels: usize, kernel_size: usize, layers: usize) -> Self {
        Self {
            channels,
            kernel_size,
            layers,
            dropout_rate: 0.0,
            eps: 1e-5,
        }
    }

    #[allow(dead_code)]
    pub fn with_dropout_rate(mut self, dropout_rate: f64) -> Self {
        self.dropout_rate = dropout_rate;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> DilatedDepthSeparableConv<B> {
        let mut convs = Vec::with_capacity(self.layers);
        for i in 0..self.layers {
            let dilation = self.kernel_size.pow(i as u32);

            let block = DDSConvBlockConfig {
                channels: self.channels,
                kernel_size: self.kernel_size,
                dilation,
                dropout_rate: self.dropout_rate,
                eps: self.eps,
            }
            .init(device);
            convs.push(block);
        }
        DilatedDepthSeparableConv { convs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::Flex;
    use burn::backend::wgpu::Wgpu;
    use burn_store::{pytorch::PytorchReader, ModuleSnapshot, PytorchStore};

    const REL_TOL: f32 = 0.05;
    const FIXTURE_DIR: &str = "tests/fixtures/vits/generator/duration_predictor/flow_step/conv/dds";
    type ForwardOutput<B> = Tensor<B, 3>;
    type TestModule<B> = DilatedDepthSeparableConv<B>;

    fn init_module<B: Backend>(device: &B::Device) -> TestModule<B> {
        DilatedDepthSeparableConvConfig::new(192, 3, 3).init(device)
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
        module.forward(
            get_tensor(input_reader, "x", device),
            get_tensor(input_reader, "x_mask", device),
            None,
        )
    }

    /// 全要素が相対誤差 `tol` 以内かをチェックする。
    /// 期待値が0に近い要素は絶対誤差(ABS_EPS)を分母の下限にして判定する。
    fn assert_allclose<B: Backend>(actual: ForwardOutput<B>, expected: &PytorchReader, tol: f32) {
        // ↓ 出力の次元数(D)もPython側 run_forward() の出力shapeに合わせて置き換える
        let actual_tensor = actual;
        let expected_tensor = get_tensor(expected, "output_0", &Default::default());
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
