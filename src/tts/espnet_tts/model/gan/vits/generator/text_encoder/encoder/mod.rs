mod layer;
mod positional_encoding;

use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::*;
use burn_store::PytorchStore;

use self::layer::{
    ConformerConvConfig, EncoderLayer, EncoderLayerConfig, MultiLayeredConv1dConfig,
    RelPositionMultiHeadedAttentionConfig,
};
use self::positional_encoding::{RelPositionalEncoding, RelPositionalEncodingConfig};

pub struct EncoderConfig {
    idim: usize,
    attention_dim: usize,
    attention_heads: usize,
    linear_units: usize,
    num_blocks: usize,
    dropout_rate: f64,
    positional_dropout_rate: f64,
    attention_dropout_rate: f64,
    input_layer: String,
    normalize_before: bool,
    concat_after: bool,
    positionwise_layer_type: String,
    positionwise_conv_kernel_size: usize,
    macaron_style: bool,
    pos_enc_layer_type: String,
    selfattention_layer_type: String,
    activation_type: String,
    use_cnn_module: bool,
    zero_triu: bool,
    cnn_module_kernel: usize,
    padding_idx: isize,
    stochastic_depth_rate: f64,
    intermediate_layers: Option<Vec<usize>>,
    ctc_softmax: String,
    conditioning_layer_dim: usize,
}
impl EncoderConfig {
    pub fn new(
        input_layer: String,
        attention_dim: usize,
        attention_heads: usize,
        linear_units: usize,
        num_blocks: usize,
        dropout_rate: f64,
        positional_dropout_rate: f64,
        attention_dropout_rate: f64,
        normalize_before: bool,
        positionwise_layer_type: String,
        positionwise_conv_kernel_size: usize,
        macaron_style: bool,
        pos_enc_layer_type: String,
        selfattention_layer_type: String,
        activation_type: String,
        use_cnn_module: bool,
        cnn_module_kernel: usize,
    ) -> Self {
        Self {
            idim: 0,
            attention_dim,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            attention_dropout_rate,
            input_layer,
            normalize_before,
            concat_after: false,
            positionwise_layer_type,
            positionwise_conv_kernel_size,
            macaron_style,
            pos_enc_layer_type,
            selfattention_layer_type,
            activation_type,
            use_cnn_module,
            zero_triu: false,
            cnn_module_kernel,
            padding_idx: -1,
            stochastic_depth_rate: 0.,
            intermediate_layers: None,
            ctc_softmax: "".to_string(),
            conditioning_layer_dim: 0,
        }
    }
    pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        let embed = RelPositionalEncodingConfig::new(self.attention_dim, self.dropout_rate).init();
        let encoders = (0..self.num_blocks)
            .map(|i| {
                EncoderLayerConfig::new(
                    self.attention_dim,
                    RelPositionMultiHeadedAttentionConfig::new(
                        self.attention_heads,
                        self.attention_dim,
                        self.attention_dropout_rate,
                    ),
                    MultiLayeredConv1dConfig::new(
                        self.attention_dim,
                        self.linear_units,
                        self.positionwise_conv_kernel_size,
                        self.dropout_rate,
                    ),
                    MultiLayeredConv1dConfig::new(
                        self.attention_dim,
                        self.linear_units,
                        self.positionwise_conv_kernel_size,
                        self.dropout_rate,
                    ),
                    ConformerConvConfig::new(
                        self.attention_dim,
                        self.cnn_module_kernel,
                        true,
                        self.activation_type.clone(),
                    ),
                    self.dropout_rate,
                    self.normalize_before,
                    self.concat_after,
                    self.stochastic_depth_rate * (1 + i) as f64 / self.num_blocks as f64,
                    self.macaron_style,
                    self.use_cnn_module,
                )
                .init(device)
            })
            .collect();
        let after_norm = LayerNormConfig::new(self.attention_dim).init(device);

        Encoder {
            embed,
            encoders,
            after_norm,
        }
    }
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    embed: RelPositionalEncoding,
    encoders: Vec<EncoderLayer<B>>,
    after_norm: LayerNorm<B>,
}
impl<B: Backend> Encoder<B> {
    pub fn forward(&self, xs: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let (xs, pos_emb) = self.embed.forward(xs);

        let mut xs = xs;
        for i in 0..self.encoders.len() {
            xs = self.encoders[i].forward(xs, pos_emb.clone(), mask.clone(), None);
        }

        let xs = self.after_norm.forward(xs);
        xs
    }

    pub fn key_remap(&self, store: PytorchStore, parents: &Vec<&str>) -> PytorchStore {
        let mut store = store;

        for (i, e) in self.encoders.iter().enumerate() {
            let mut parents = parents.clone();
            let i = i.to_string();
            parents.push("encoders");
            parents.push(i.as_str());
            store = e.key_remap(store, &parents);
        }
        store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Flex, Wgpu};
    use burn_store::{pytorch::PytorchReader, ModuleSnapshot, PytorchStore};

    const REL_TOL: f32 = 0.05;
    const FIXTURE_DIR: &str = "tests/fixtures/vits/generator/text_encoder/encoder/mod";
    type ForwardOutput<B> = Tensor<B, 3>;
    type TestModule<B> = Encoder<B>;

    fn init_module<B: Backend>(device: &B::Device) -> TestModule<B> {
        EncoderConfig::new(
            "".to_string(),
            192,
            2,
            768,
            6,
            0.1,
            0.0,
            0.1,
            true,
            "conv1d".to_string(),
            3,
            true,
            "rel_pos".to_string(),
            "rel_selfattn".to_string(),
            "swish".to_string(),
            false,
            0,
        )
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
        module.forward(
            get_tensor(input_reader, "xs", device),
            get_tensor(input_reader, "masks", device),
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
