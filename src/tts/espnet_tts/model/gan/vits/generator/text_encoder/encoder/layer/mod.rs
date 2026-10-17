mod attention;
mod conformer_conv;
mod multi_layered_conv;

use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig};
use burn::prelude::*;
use burn_store::PytorchStore;

use self::attention::RelPositionMultiHeadedAttention;
pub use self::attention::RelPositionMultiHeadedAttentionConfig;
use self::conformer_conv::ConformerConv;
pub use self::conformer_conv::ConformerConvConfig;
use self::multi_layered_conv::MultiLayeredConv1d;
pub use self::multi_layered_conv::MultiLayeredConv1dConfig;

pub struct EncoderLayerConfig {
    size: usize,
    self_attn: RelPositionMultiHeadedAttentionConfig,
    feed_forward: MultiLayeredConv1dConfig,
    feed_forward_macaron: MultiLayeredConv1dConfig,
    conv_module: ConformerConvConfig,
    dropout_rate: f64,
    normalize_before: bool,
    concat_after: bool,
    stochastic_depth_rate: f64,
    macaron_style: bool,
    use_cnn_module: bool,
}
impl EncoderLayerConfig {
    pub fn new(
        size: usize,
        self_attn: RelPositionMultiHeadedAttentionConfig,
        feed_forward: MultiLayeredConv1dConfig,
        feed_forward_macaron: MultiLayeredConv1dConfig,
        conv_module: ConformerConvConfig,
        dropout_rate: f64,
        normalize_before: bool,
        concat_after: bool,
        stochastic_depth_rate: f64,
        macaron_style: bool,
        use_cnn_module: bool,
    ) -> Self {
        Self {
            size,
            self_attn,
            feed_forward,
            feed_forward_macaron,
            conv_module,
            dropout_rate,
            normalize_before,
            concat_after,
            stochastic_depth_rate,
            macaron_style,
            use_cnn_module,
        }
    }
    pub fn init<B: Backend>(&self, device: &B::Device) -> EncoderLayer<B> {
        assert!(self.macaron_style, "macaron_style must be true");
        assert!(
            self.stochastic_depth_rate <= 0.0,
            "stochastic_depth_rate must be 0.0"
        );
        let (norm_conv, conv_module, norm_final) = if self.use_cnn_module {
            (
                Some(LayerNormConfig::new(self.size).init(device)),
                Some(self.conv_module.init(device)),
                Some(LayerNormConfig::new(self.size).init(device)),
            )
        } else {
            (None, None, None)
        };
        EncoderLayer {
            self_attn: self.self_attn.init(device),
            norm_ff_macaron: LayerNormConfig::new(self.size).init(device),
            feed_forward_macaron: self.feed_forward_macaron.init(device),
            norm_mha: LayerNormConfig::new(self.size).init(device),
            dropout: DropoutConfig::new(self.dropout_rate).init(),
            norm_conv: norm_conv,
            conv_module: conv_module,
            feed_forward: self.feed_forward.init(device),
            norm_ff: LayerNormConfig::new(self.size).init(device),
            norm_final,
            ff_scale: 0.5,
            normalize_before: self.normalize_before,
            concat_after: self.concat_after,
        }
    }
}

#[derive(Module, Debug)]
pub struct EncoderLayer<B: Backend> {
    self_attn: RelPositionMultiHeadedAttention<B>,
    norm_ff_macaron: LayerNorm<B>,
    feed_forward_macaron: MultiLayeredConv1d<B>,
    norm_mha: LayerNorm<B>,
    dropout: Dropout,
    norm_conv: Option<LayerNorm<B>>,
    conv_module: Option<ConformerConv<B>>,
    feed_forward: MultiLayeredConv1d<B>,
    norm_ff: LayerNorm<B>,
    norm_final: Option<LayerNorm<B>>,
    ff_scale: f64,
    normalize_before: bool,
    concat_after: bool,
}
impl<B: Backend> EncoderLayer<B> {
    /// # Args
    /// * `x` - (batch, time, size)
    /// * `pos_emb` - (1, time, size)
    /// * `mask` - (batch, 1, time)
    /// * `cache` - (batch, time - 1, size)
    /// # Returns
    /// (batch, time, size)
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        pos_emb: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        cache: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        //EncoderOutput<B> {
        if self.normalize_before {
            self.forawrd_norm_before(x, pos_emb, mask, cache)
        } else {
            panic!("Not impl")
        }
    }

    fn forawrd_norm_before(
        &self,
        x: Tensor<B, 3>,
        pos_emb: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        cache: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        //EncoderOutput<B> {
        let stoch_layer_coeff = 1.0;
        // macaron style
        let residual = x.clone();
        let x = self.norm_ff_macaron.forward(x);
        let x = self.feed_forward_macaron.forward(x, None);
        let x = self.dropout.forward(x);
        let x = residual + stoch_layer_coeff * self.ff_scale * x;

        // multi-head self-attention
        let residual = x.clone();
        let x = self.norm_mha.forward(x);

        let x_q = if cache.is_none() {
            x.clone()
        } else {
            panic!("Not impl.")
        };
        let x_att = self
            .self_attn
            .forward(x_q, x.clone(), x.clone(), pos_emb, mask.clone());

        let x = if self.concat_after {
            panic!("Not impl.")
        } else {
            residual + stoch_layer_coeff * self.dropout.forward(x_att)
        };

        // conv module
        let x = match self.conv_module.as_ref() {
            Some(conv_module) => {
                let residual = x.clone();
                let x = self.norm_conv.as_ref().unwrap().forward(x);
                let x_conv = conv_module.forward(x, mask.clone());
                residual + stoch_layer_coeff * self.dropout.forward(x_conv)
            }
            None => x,
        };

        // feed forward
        let residual = x.clone();
        let x = self.norm_ff.forward(x);
        let x = self.feed_forward.forward(x, None);
        let x = self.dropout.forward(x);
        let x = residual + stoch_layer_coeff * self.ff_scale * x;

        // self.conv_module があるときに正規化
        let x = match self.norm_final {
            Some(ref norm_final) => norm_final.forward(x),
            None => x,
        };

        // append cache
        if cache.is_some() {
            panic!("Not impl");
        }

        // EncoderOutput { output: x, mask }
        x
    }

    pub fn key_remap(&self, store: PytorchStore, parents: &Vec<&str>) -> PytorchStore {
        let mut parents = parents.clone();
        let store = match self.conv_module.as_ref() {
            Some(conv_module) => {
                parents.push("conv_module");
                let s = conv_module.key_remap(store, &parents);
                parents.pop();
                s
            }
            None => store,
        };
        store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::Flex;
    use burn::backend::wgpu::Wgpu;
    use burn_store::{pytorch::PytorchReader, ModuleSnapshot, PytorchStore};

    const REL_TOL: f32 = 0.05;
    const FIXTURE_DIR: &str = "tests/fixtures/vits/generator/text_encoder/encoder/layer/mod";
    type ForwardOutput<B> = Tensor<B, 3>;
    type TestModule<B> = EncoderLayer<B>;

    fn init_module<B: Backend>(device: &B::Device) -> TestModule<B> {
        let attention_dim = 192;
        let attention_heads = 2;
        let attention_dropout_rate = 0.1;
        let linear_units = 768;
        let positionwise_conv_kernel_size = 3;
        let dropout_rate = 0.1;
        let cnn_module_kernel = 0;
        let activation_type = "swish".to_string();
        let normalize_before = true;
        EncoderLayerConfig::new(
            attention_dim,
            RelPositionMultiHeadedAttentionConfig::new(
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            ),
            MultiLayeredConv1dConfig::new(
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            ),
            MultiLayeredConv1dConfig::new(
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            ),
            ConformerConvConfig::new(
                attention_dim,
                cnn_module_kernel,
                true,
                activation_type.clone(),
            ),
            dropout_rate,
            normalize_before,
            false,
            0.0,
            true,
            false,
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
            get_tensor(input_reader, "x_input.0", device),
            get_tensor(input_reader, "x_input.1", device),
            get_tensor(input_reader, "mask", device),
            None,
        )
    }

    /// 全要素が相対誤差 `tol` 以内かをチェックする。
    /// 期待値が0に近い要素は絶対誤差(ABS_EPS)を分母の下限にして判定する。
    fn assert_allclose<B: Backend>(actual: ForwardOutput<B>, expected: &PytorchReader, tol: f32) {
        // ↓ 出力の次元数(D)もPython側 run_forward() の出力shapeに合わせて置き換える
        let actual_output_0 = actual;
        let expected_output_0 = get_tensor(expected, "output_0.0", &Default::default());

        assert_close(actual_output_0, expected_output_0, tol);
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
