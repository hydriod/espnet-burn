mod decoder;
mod duration_predictor;
mod flow;
// mod posterior_encoder;
mod text_encoder;

use burn::prelude::*;
use burn::tensor::Distribution;
use burn_store::PytorchStore;

use crate::{error::Error, tts::config::GeneratorParams};

use self::{
    decoder::VITSDecoder,
    duration_predictor::{StochasticDurationPredictor, StochasticDurationPredictorConfig},
    flow::VITSFlow,
    // posterior_encoder::PosteriorEncoder,
    text_encoder::{TextEncoder, TextEncoderConfig},
};

impl GeneratorParams {
    pub fn init_vits_generator<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<VITSGenerator<B>, Error> {
        // Ported from `VITSGenerator.__init__` in
        // `espnet2/gan_tts/vits/generator.py`.
        let hidden_channels = self.hidden_channels as usize;
        // `-1` (Python default for "unused") means "no conditioning".
        let global_channels = self.global_channels.max(0) as usize;

        let text_encoder = TextEncoderConfig::new(
            self.vocabs,
            self.hidden_channels,
            self.text_encoder_attention_heads,
            self.hidden_channels * self.text_encoder_ffn_expand,
            self.text_encoder_blocks,
            self.text_encoder_positionwise_layer_type.clone(),
            self.text_encoder_positionwise_conv_kernel_size,
            self.text_encoder_positional_encoding_layer_type.clone(),
            self.text_encoder_self_attention_layer_type.clone(),
            self.text_encoder_activation_type.clone(),
            self.text_encoder_normalize_before,
            self.text_encoder_dropout_rate,
            self.text_encoder_positional_dropout_rate,
            self.text_encoder_attention_dropout_rate,
            // Unused when `use_conformer_conv_in_text_encoder` is false
            // (the test config passes `-1`).
            self.text_encoder_conformer_kernel_size.max(0),
            self.use_macaron_style_in_text_encoder,
            self.use_conformer_conv_in_text_encoder,
        )?
        .init(device);
        let duration_predictor = StochasticDurationPredictorConfig::new(
            hidden_channels,
            self.stochastic_duration_predictor_kernel_size as usize,
            self.stochastic_duration_predictor_dropout_rate,
            self.stochastic_duration_predictor_flows as usize,
            self.stochastic_duration_predictor_dds_conv_layers as usize,
            global_channels,
        )
        .init(device);
        let flow = VITSFlow::init_residual_coupling(
            hidden_channels,
            hidden_channels,
            self.flow_flows as usize,
            self.flow_kernel_size as usize,
            self.flow_base_dilation as usize,
            self.flow_layers as usize,
            self.global_channels as isize,
            self.flow_dropout_rate,
            self.use_weight_norm_in_flow,
            self.use_only_mean_in_flow,
            device,
        )?;
        let decoder = VITSDecoder::init_hifi_gan(
            hidden_channels,
            1,
            self.decoder_channels as usize,
            global_channels,
            self.decoder_kernel_size as usize,
            self.decoder_upsample_scales
                .iter()
                .map(|&s| s as usize)
                .collect(),
            self.decoder_upsample_kernel_sizes
                .iter()
                .map(|&s| s as usize)
                .collect(),
            self.decoder_resblock_kernel_sizes
                .iter()
                .map(|&s| s as usize)
                .collect(),
            self.decoder_resblock_dilations
                .iter()
                .map(|dils| dils.iter().map(|&d| d as usize).collect())
                .collect(),
            device,
        )?;
        Ok(VITSGenerator {
            text_encoder,
            duration_predictor,
            flow,
            decoder,
        })
    }
}

#[derive(Module, Debug)]
pub struct VITSGenerator<B: Backend> {
    text_encoder: TextEncoder<B>,
    duration_predictor: StochasticDurationPredictor<B>,
    // posterior_encoder: PosteriorEncoder<B>,
    flow: VITSFlow<B>,
    decoder: VITSDecoder<B>,
}
impl<B: Backend> VITSGenerator<B> {
    /// Run inference (non teacher-forcing path of `VITSGenerator.inference`).
    ///
    /// Speaker / language conditioning (`sids`, `spembs`, `lids`) is not
    /// supported yet: the test config uses `spks = -1` / `global_channels = -1`,
    /// so `g` is always `None` as in the Python code path.
    pub fn inference(
        &self,
        text: Tensor<B, 2, Int>,
        text_length: Tensor<B, 1>,
        options: VITSSynthesisOptions,
        use_teacher_forcing: bool,
    ) -> Result<VITSGenOutput<B>, Error> {
        if use_teacher_forcing {
            return Err(Error::Argument(
                "use_teacher_forcing=true requires the posterior encoder, which is not ported yet."
                    .to_string(),
            ));
        }
        let device = text.device();
        let result = self
            .text_encoder
            .forward(text, text_length)
            .map_err(|e| Error::Other(e.to_string()))?;
        let x = result.encoded;
        let m_p = result.mean;
        let logs_p = result.scale;
        let x_mask = result.mask;

        // Duration prediction with the inverse flow.
        let logw = self.duration_predictor.forward(
            x,
            x_mask.clone(),
            None,
            None,
            true,
            options.noise_scale_dur,
        )?;
        let w = logw.exp() * x_mask.clone() * options.alpha;
        let dur = w.ceil();

        // Monotonic alignment from the predicted durations.
        let y_lengths = dur.clone().sum_dims_squeeze(&[1, 2]).clamp_min(1.0).int();
        let y_mask = make_non_pad_mask(y_lengths, &device).unsqueeze_dim(1);
        let attn_mask = x_mask.clone().unsqueeze_dim(2) * y_mask.clone().unsqueeze_dim(3);
        let attn = Self::generate_path(dur.clone(), attn_mask);

        // Expand text statistics to the feature length:
        // (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats).
        let attn_squeezed = attn.clone().squeeze_dim(1);
        let m_p = attn_squeezed
            .clone()
            .matmul(m_p.swap_dims(1, 2))
            .swap_dims(1, 2);
        let logs_p = attn_squeezed.matmul(logs_p.swap_dims(1, 2)).swap_dims(1, 2);

        // Sample from the prior and run the inverse flow + decoder.
        let z_p = m_p.clone()
            + m_p.random_like(Distribution::Normal(0.0, 1.0)) * logs_p.exp() * options.noise_scale;
        let z = self.flow.inverse(z_p, y_mask.clone(), None);
        let wav = self.decoder.forward(z * y_mask);

        Ok(VITSGenOutput {
            generated: wav.squeeze_dim(1),
            attention_weight: attn.squeeze_dim(1),
            duration: dur.squeeze_dim(1),
        })
    }

    /// Generate a monotonic attention path from durations.
    ///
    /// Ported from `VITSGenerator._generate_path`:
    /// `path - F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]`, transposed to
    /// `(B, 1, T_feats, T_text)` and masked.
    ///
    /// # Args
    /// - `dur`: Duration tensor (B, 1, T_text).
    /// - `mask`: Attention mask tensor (B, 1, T_feats, T_text).
    fn generate_path(dur: Tensor<B, 3>, mask: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, _, t_y, t_x] = mask.dims();
        let device = dur.device();
        // Cumulative durations, flattened to (B * T_x,).
        let cum_dur_flat: Tensor<B, 1> = dur.cumsum(2).reshape([b * t_x]);
        // (1, T_y) < (B * T_x, 1) -> (B * T_x, T_y) -> (B, T_x, T_y).
        let steps = Tensor::<B, 1, Int>::arange(0..t_y as i64, &device).float();
        let steps: Tensor<B, 2> = steps.unsqueeze_dim(0);
        let thresh: Tensor<B, 2> = cum_dur_flat.unsqueeze_dim(1);
        let path: Tensor<B, 3> = steps.lower(thresh).float().reshape([b, t_x, t_y]);
        // Shift along T_x by one (front-padded with zeros).
        let shifted = if t_x > 1 {
            Tensor::cat(
                vec![
                    Tensor::zeros([b, 1, t_y], &device),
                    path.clone().slice_dim(1, 0..t_x - 1),
                ],
                1,
            )
        } else {
            Tensor::zeros([b, t_x, t_y], &device)
        };
        (path - shifted).unsqueeze_dim(1).swap_dims(2, 3) * mask
    }

    pub fn key_remap(&mut self, store: PytorchStore, parents: &Vec<&str>) -> PytorchStore {
        // NOTE: duration_predictor must run before decoder/flow: its DDS
        // key_remap disables `map_indices_contiguous` on the shared store,
        // which the DDS weight layout requires (raw keys like
        // `dds.convs.0.2` match the Burn module paths as-is). The HiFiGAN
        // loader below probes both mapped (`.0`) and raw (`.1`) key
        // variants, so it works with either cache state.
        let mut parents = parents.clone();
        let p = "text_encoder";
        parents.push(p);
        let store = self.text_encoder.key_remap(store, &parents);
        parents.pop();

        let p = "duration_predictor";
        parents.push(p);
        let store = self.duration_predictor.key_remap(store, &parents);
        parents.pop();

        let p = "decoder";
        parents.push(p);
        let store = self.decoder.key_remap(store, &parents);
        parents.pop();

        let p = "flow";
        parents.push(p);
        let store = self.flow.key_remap(store, &parents);
        parents.pop();

        store
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VITSSynthesisOptions {
    /// speed control alpha
    pub alpha: f64,
    pub noise_scale: f64,
    pub noise_scale_dur: f64,
}
impl Default for VITSSynthesisOptions {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            noise_scale: 0.667,
            noise_scale_dur: 0.8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VITSGenOutput<B: Backend> {
    pub generated: Tensor<B, 2>,
    pub attention_weight: Tensor<B, 3>,
    pub duration: Tensor<B, 2>,
}

/// Build a non-padding mask (B, T_max) from lengths (B,).
///
/// Mirrors `espnet2.legacy.nets.pytorch_backend.nets_utils.make_non_pad_mask`.
fn make_non_pad_mask<B: Backend>(lengths: Tensor<B, 1, Int>, device: &B::Device) -> Tensor<B, 2> {
    let lens: Vec<i32> = lengths
        .to_data()
        .to_vec::<i32>()
        .expect("failed to read lengths from tensor");
    let max_len = lens.iter().copied().max().unwrap_or(0).max(0) as usize;
    let rows = lens
        .iter()
        .map(|&len| {
            let len = (len.max(0) as usize).min(max_len);
            Tensor::cat(
                vec![
                    Tensor::<B, 1>::ones([len], device),
                    Tensor::<B, 1>::zeros([max_len - len], device),
                ],
                0,
            )
        })
        .collect::<Vec<Tensor<B, 1>>>();
    Tensor::stack::<2>(rows, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::Flex;
    use burn::backend::wgpu::Wgpu;
    use burn_store::{pytorch::PytorchReader, ModuleSnapshot, PytorchStore};

    const REL_TOL: f32 = 0.05;
    const FIXTURE_DIR: &str = "tests/fixtures/vits/generator/mod";
    type ForwardOutput<B> = VITSGenOutput<B>;
    type TestModule<B> = VITSGenerator<B>;

    fn init_module<B: Backend>(device: &B::Device) -> TestModule<B> {
        GeneratorParams {
            hidden_channels: 192,
            spks: -1,
            global_channels: -1,
            segment_size: 32,
            text_encoder_attention_heads: 2,
            text_encoder_ffn_expand: 4,
            text_encoder_blocks: 6,
            text_encoder_positionwise_layer_type: "conv1d".to_string(),
            text_encoder_positionwise_conv_kernel_size: 3,
            text_encoder_positional_encoding_layer_type: "rel_pos".to_string(),
            text_encoder_self_attention_layer_type: "rel_selfattn".to_string(),
            text_encoder_activation_type: "swish".to_string(),
            text_encoder_normalize_before: true,
            text_encoder_dropout_rate: 0.1,
            text_encoder_positional_dropout_rate: 0.0,
            text_encoder_attention_dropout_rate: 0.1,
            use_macaron_style_in_text_encoder: true,
            use_conformer_conv_in_text_encoder: false,
            text_encoder_conformer_kernel_size: -1,
            decoder_kernel_size: 7,
            decoder_channels: 512,
            decoder_upsample_scales: vec![8, 8, 2, 2, 2],
            decoder_upsample_kernel_sizes: vec![16, 16, 4, 4, 4],
            decoder_resblock_kernel_sizes: vec![3, 7, 11],
            decoder_resblock_dilations: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            use_weight_norm_in_decoder: true,
            posterior_encoder_kernel_size: 5,
            posterior_encoder_layers: 16,
            posterior_encoder_stacks: 1,
            posterior_encoder_base_dilation: 1,
            posterior_encoder_dropout_rate: 0.0,
            use_weight_norm_in_posterior_encoder: true,
            flow_flows: 4,
            flow_kernel_size: 5,
            flow_base_dilation: 1,
            flow_layers: 4,
            flow_dropout_rate: 0.0,
            use_weight_norm_in_flow: true,
            use_only_mean_in_flow: true,
            stochastic_duration_predictor_kernel_size: 3,
            stochastic_duration_predictor_dropout_rate: 0.5,
            stochastic_duration_predictor_flows: 4,
            stochastic_duration_predictor_dds_conv_layers: 3,
            vocabs: 47,
            aux_channels: 1025,
        }
        .init_vits_generator(device)
        .unwrap()
    }

    fn key_remap<B: Backend>(store: PytorchStore, module: &mut TestModule<B>) -> PytorchStore {
        let store = module.key_remap(store, &vec![]);
        store
    }

    fn forward<B: Backend>(
        module: &TestModule<B>,
        input_reader: &PytorchReader,
        device: &B::Device,
    ) -> ForwardOutput<B> {
        // forward内で生成される乱数の影響を消すためnoise_scaleを0.0に設定
        module
            .inference(
                get_tensor(input_reader, "text", device).int(),
                get_tensor(input_reader, "text_lengths", device),
                VITSSynthesisOptions {
                    alpha: 1.0,
                    noise_scale: 0.0,
                    noise_scale_dur: 0.0,
                },
                false, // use_teacher_forcingは常にfalse
            )
            .unwrap_or_else(|e| panic!("forwardに失敗。\n{e}"))
    }

    /// 全要素が相対誤差 `tol` 以内かをチェックする。
    /// 期待値が0に近い要素は絶対誤差(ABS_EPS)を分母の下限にして判定する。
    fn assert_allclose<B: Backend>(actual: ForwardOutput<B>, expected: &PytorchReader, tol: f32) {
        // ↓ 出力の次元数(D)もPython側 run_forward() の出力shapeに合わせて置き換える
        let actual_tensor = actual.generated;
        let expected_tensor = get_tensor(expected, "output_0", &Default::default());
        assert_close(actual_tensor, expected_tensor, tol);

        let actual_tensor = actual.attention_weight;
        let expected_tensor = get_tensor(expected, "output_1", &Default::default());
        assert_close(actual_tensor, expected_tensor, tol);

        let actual_tensor = actual.duration;
        let expected_tensor = get_tensor(expected, "output_2", &Default::default());
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
        let mut weight_store = key_remap(weight_store, &mut module);
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
