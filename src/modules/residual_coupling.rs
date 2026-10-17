//! Residual affine coupling block (flow) in VITS.
//!
//! Ported from `espnet2/gan_tts/vits/residual_coupling.py`
//! (`ResidualAffineCouplingBlock`, `ResidualAffineCouplingLayer`)
//! and `espnet2/gan_tts/vits/flow.py` (`FlipFlow`).
//! Original copyright 2021 Tomoki Hayashi, Apache 2.0.
//! This code is based on <https://github.com/jaywalnut310/vits>.
//!
//! Notes on the port:
//! - `torch.nn.ModuleList` alternating
//!   `[CouplingLayer, FlipFlow] * flows` is a plain
//!   `Vec<ResidualAffineCouplingLayer>`; the channel flip
//!   (`torch.flip(x, [1])`) is applied inline in [`ResidualAffineCouplingBlock::forward`].
//! - `use_weight_norm` is accepted for API compatibility but is a no-op:
//!   Burn has no weight-norm reparameterization. Checkpoints saved with
//!   `use_weight_norm=True` store `*.weight_g` / `*.weight_v` instead of
//!   `*.weight`; [`ResidualAffineCouplingBlock::key_remap`] fuses them into
//!   plain weights (`w = g * v / ||v||`), matching
//!   `torch.nn.utils.weight_norm` semantics.
//! - `global_channels <= 0` (Python default `-1`) disables the global
//!   conditioning branch, matching the Python `if global_channels > 0` check.

use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::prelude::*;
use burn_store::{ModuleStore, PytorchStore};

use crate::error::Error;
use crate::modules::wavenet::{WaveNet, WaveNetConfig};

// ---------------------------------------------------------------------------
// Residual affine coupling layer
// ---------------------------------------------------------------------------

/// Configuration for [`ResidualAffineCouplingLayer`].
///
/// Mirrors `ResidualAffineCouplingLayer.__init__` in `residual_coupling.py`.
#[derive(Clone, Debug)]
pub struct ResidualAffineCouplingLayerConfig {
    /// Number of input channels. Must be even (split into two halves).
    pub in_channels: usize,
    /// Number of hidden channels.
    pub hidden_channels: usize,
    /// Kernel size of the WaveNet encoder.
    pub kernel_size: usize,
    /// Base dilation factor of the WaveNet encoder.
    pub base_dilation: usize,
    /// Number of layers of the WaveNet encoder.
    pub layers: usize,
    /// Number of stacks of the WaveNet encoder.
    pub stacks: usize,
    /// Number of global-conditioning channels. `<= 0` disables it.
    pub global_channels: isize,
    /// Dropout rate of the WaveNet encoder.
    pub dropout_rate: f64,
    /// Accepted for API compatibility; no-op in this Burn port.
    pub use_weight_norm: bool,
    /// Whether to use bias in the WaveNet encoder.
    pub bias: bool,
    /// Whether to estimate only the mean (`true`) or mean and log-scale
    /// (`false`) in the projection layer.
    pub use_only_mean: bool,
}

impl Default for ResidualAffineCouplingLayerConfig {
    fn default() -> Self {
        Self {
            in_channels: 192,
            hidden_channels: 192,
            kernel_size: 5,
            base_dilation: 1,
            layers: 5,
            stacks: 1,
            global_channels: -1,
            dropout_rate: 0.0,
            use_weight_norm: true,
            bias: true,
            use_only_mean: true,
        }
    }
}

impl ResidualAffineCouplingLayerConfig {
    pub fn new(in_channels: usize, hidden_channels: usize) -> Self {
        Self {
            in_channels,
            hidden_channels,
            ..Default::default()
        }
    }

    pub fn with_kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = kernel_size;
        self
    }

    pub fn with_base_dilation(mut self, base_dilation: usize) -> Self {
        self.base_dilation = base_dilation;
        self
    }

    pub fn with_layers(mut self, layers: usize) -> Self {
        self.layers = layers;
        self
    }

    pub fn with_stacks(mut self, stacks: usize) -> Self {
        self.stacks = stacks;
        self
    }

    pub fn with_global_channels(mut self, global_channels: isize) -> Self {
        self.global_channels = global_channels;
        self
    }

    pub fn with_dropout_rate(mut self, dropout_rate: f64) -> Self {
        self.dropout_rate = dropout_rate;
        self
    }

    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_use_only_mean(mut self, use_only_mean: bool) -> Self {
        self.use_only_mean = use_only_mean;
        self
    }

    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<ResidualAffineCouplingLayer<B>, Error> {
        if self.in_channels % 2 != 0 {
            return Err(Error::Argument(
                "in_channels should be divisible by 2.".to_string(),
            ));
        }
        let half_channels = self.in_channels / 2;

        let input_conv = Conv1dConfig::new(half_channels, self.hidden_channels, 1)
            .with_bias(true)
            .init(device);

        // Matches the `WaveNet(...)` construction in
        // `ResidualAffineCouplingLayer.__init__`.
        let encoder = WaveNetConfig::new()
            .with_in_channels(0)
            .with_out_channels(0)
            .with_kernel_size(self.kernel_size)
            .with_layers(self.layers)
            .with_stacks(self.stacks)
            .with_base_dilation(self.base_dilation)
            .with_residual_channels(self.hidden_channels)
            .with_aux_channels(-1)
            .with_gate_channels(self.hidden_channels * 2)
            .with_skip_channels(self.hidden_channels)
            .with_global_channels(self.global_channels)
            .with_dropout_rate(self.dropout_rate)
            .with_bias(self.bias)
            .with_use_first_conv(false)
            .with_use_last_conv(false)
            .with_scale_residual(false)
            .with_scale_skip_connect(true)
            .init(device)?;

        let proj_out_channels = if self.use_only_mean {
            half_channels
        } else {
            half_channels * 2
        };
        let proj = Conv1dConfig::new(self.hidden_channels, proj_out_channels, 1)
            .with_bias(true)
            .init(device);

        Ok(ResidualAffineCouplingLayer {
            input_conv,
            encoder,
            proj,
            half_channels,
            use_only_mean: self.use_only_mean,
        })
    }
}

/// Residual affine coupling layer.
///
/// Corresponds to `ResidualAffineCouplingLayer` in `residual_coupling.py`.
#[derive(Module, Debug)]
pub struct ResidualAffineCouplingLayer<B: Backend> {
    input_conv: Conv1d<B>,
    encoder: WaveNet<B>,
    proj: Conv1d<B>,
    half_channels: usize,
    use_only_mean: bool,
}

/// Single coupling layer as stored in [`ResidualAffineCouplingBlock`].
///
/// The extra `layer` path level keeps Burn parameter paths disjoint from
/// the Python checkpoint layout: Python interleaves each coupling layer
/// with a parameter-free flip (`[layer, flip] * flows`), so its
/// `ModuleList` positions (`flows.0`, `flows.2`, ...) textually collide
/// with Burn's contiguous `Vec` indices (`flows.0`, `flows.1`, ...).
/// Without this level, a follow-up `load_from` call would silently apply
/// one layer's weights to another (e.g. Burn layer 2 reading Python
/// `flows.2`, which holds layer 1). All parameters are therefore assigned
/// by [`ResidualAffineCouplingBlock::key_remap`] (which resolves both raw
/// and renumbered spellings); `load_from` finds no matching paths and
/// leaves them untouched.
#[derive(Module, Debug)]
struct FlowLayer<B: Backend> {
    layer: ResidualAffineCouplingLayer<B>,
}

impl<B: Backend> ResidualAffineCouplingLayer<B> {
    /// Calculate forward propagation.
    ///
    /// # Args
    /// - `x`: Input tensor (B, in_channels, T).
    /// - `x_mask`: Mask tensor (B, 1, T).
    /// - `g`: Global conditioning tensor (B, global_channels, 1).
    /// - `inverse`: Whether to run the inverse flow.
    ///
    /// # Returns
    /// - Output tensor (B, in_channels, T).
    /// - Log-determinant tensor (B, 1, 1) for NLL if not inverse, else `None`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
        inverse: bool,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 3>>) {
        let parts = x.chunk(2, 1);
        let (xa, xb) = (parts[0].clone(), parts[1].clone());

        let h = self.input_conv.forward(xa.clone()) * x_mask.clone();
        let h = self
            .encoder
            .forward(h, Some(x_mask.clone()), None, g.clone());
        let stats = self.proj.forward(h) * x_mask.clone();

        let (m, logs) = if self.use_only_mean {
            (stats.clone(), Tensor::zeros_like(&stats))
        } else {
            let halves = stats.chunk(2, 1);
            (halves[0].clone(), halves[1].clone())
        };

        if !inverse {
            let xb = m.clone() + xb * logs.clone().exp() * x_mask.clone();
            let x = Tensor::cat(vec![xa, xb], 1);
            // Sum over channel and time dims, keeping dims: (B, 1, 1).
            let logdet = logs.sum_dims(&[1, 2]);
            (x, Some(logdet))
        } else {
            let xb = (xb - m) * logs.mul_scalar(-1.0).exp() * x_mask.clone();
            (Tensor::cat(vec![xa, xb], 1), None)
        }
    }
}

// ---------------------------------------------------------------------------
// Residual affine coupling block
// ---------------------------------------------------------------------------

/// Configuration for [`ResidualAffineCouplingBlock`].
///
/// Mirrors `ResidualAffineCouplingBlock.__init__` in `residual_coupling.py`.
/// Note the Python default `layers=4` (the layer default is `layers=5`).
#[derive(Clone, Debug)]
pub struct ResidualAffineCouplingBlockConfig {
    /// Number of input channels. Must be even.
    pub in_channels: usize,
    /// Number of hidden channels.
    pub hidden_channels: usize,
    /// Number of coupling layers (each followed by a flip, as in Python).
    pub flows: usize,
    /// Kernel size of the WaveNet encoder in each layer.
    pub kernel_size: usize,
    /// Base dilation factor of the WaveNet encoder in each layer.
    pub base_dilation: usize,
    /// Number of layers of the WaveNet encoder in each layer.
    pub layers: usize,
    /// Number of global-conditioning channels. `<= 0` disables it.
    pub global_channels: isize,
    /// Dropout rate of the WaveNet encoder in each layer.
    pub dropout_rate: f64,
    /// Accepted for API compatibility; no-op in this Burn port.
    pub use_weight_norm: bool,
    /// Whether to use bias in the WaveNet encoder of each layer.
    pub bias: bool,
    /// Whether to estimate only the mean in each layer.
    pub use_only_mean: bool,
}

impl Default for ResidualAffineCouplingBlockConfig {
    fn default() -> Self {
        Self {
            in_channels: 192,
            hidden_channels: 192,
            flows: 4,
            kernel_size: 5,
            base_dilation: 1,
            layers: 4,
            global_channels: -1,
            dropout_rate: 0.0,
            use_weight_norm: true,
            bias: true,
            use_only_mean: true,
        }
    }
}

impl ResidualAffineCouplingBlockConfig {
    pub fn new(in_channels: usize, hidden_channels: usize, flows: usize) -> Self {
        Self {
            in_channels,
            hidden_channels,
            flows,
            ..Default::default()
        }
    }

    pub fn with_kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = kernel_size;
        self
    }

    pub fn with_base_dilation(mut self, base_dilation: usize) -> Self {
        self.base_dilation = base_dilation;
        self
    }

    pub fn with_layers(mut self, layers: usize) -> Self {
        self.layers = layers;
        self
    }

    pub fn with_global_channels(mut self, global_channels: isize) -> Self {
        self.global_channels = global_channels;
        self
    }

    pub fn with_dropout_rate(mut self, dropout_rate: f64) -> Self {
        self.dropout_rate = dropout_rate;
        self
    }

    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_use_only_mean(mut self, use_only_mean: bool) -> Self {
        self.use_only_mean = use_only_mean;
        self
    }

    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<ResidualAffineCouplingBlock<B>, Error> {
        if self.in_channels % 2 != 0 {
            return Err(Error::Argument(
                "in_channels should be divisible by 2.".to_string(),
            ));
        }
        // Python stacks is hardcoded to 1 in the block constructor.
        let layer_config = ResidualAffineCouplingLayerConfig {
            in_channels: self.in_channels,
            hidden_channels: self.hidden_channels,
            kernel_size: self.kernel_size,
            base_dilation: self.base_dilation,
            layers: self.layers,
            stacks: 1,
            global_channels: self.global_channels,
            dropout_rate: self.dropout_rate,
            use_weight_norm: self.use_weight_norm,
            bias: self.bias,
            use_only_mean: self.use_only_mean,
        };
        let mut flows = Vec::with_capacity(self.flows);
        for _ in 0..self.flows {
            flows.push(FlowLayer {
                layer: layer_config.init(device)?,
            });
        }
        Ok(ResidualAffineCouplingBlock { flows })
    }
}

/// Residual affine coupling block module.
///
/// Corresponds to `ResidualAffineCouplingBlock` in `residual_coupling.py`,
/// used as "Flow" in VITS. Each coupling layer is followed by a channel flip
/// (`FlipFlow` in `flow.py`), matching the Python
/// `self.flows += [ResidualAffineCouplingLayer(...), FlipFlow()]` pattern.
#[derive(Module, Debug)]
pub struct ResidualAffineCouplingBlock<B: Backend> {
    flows: Vec<FlowLayer<B>>,
}

impl<B: Backend> ResidualAffineCouplingBlock<B> {
    /// Calculate forward propagation.
    ///
    /// # Args
    /// - `x`: Input tensor (B, in_channels, T).
    /// - `x_mask`: Mask tensor (B, 1, T).
    /// - `g`: Global conditioning tensor (B, global_channels, 1).
    /// - `inverse`: Whether to run the inverse flow.
    ///
    /// # Returns
    /// - Output tensor (B, in_channels, T).
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
        inverse: bool,
    ) -> Tensor<B, 3> {
        if !inverse {
            let mut x = x;
            for flow in &self.flows {
                let (x_next, _) = flow.layer.forward(x, x_mask.clone(), g.clone(), false);
                // FlipFlow: `torch.flip(x, [1])`.
                x = x_next.flip([1]);
            }
            x
        } else {
            let mut x = x;
            // `reversed(self.flows)`: flip first, then the inverse layer.
            for flow in self.flows.iter().rev() {
                x = x.flip([1]);
                let (x_next, _) = flow.layer.forward(x, x_mask.clone(), g.clone(), true);
                x = x_next;
            }
            x
        }
    }

    /// Fuse weight-norm parameters of the nested layers.
    ///
    /// Checkpoints saved with `use_weight_norm=True` (the ESPnet default)
    /// store `*.weight_g` and `*.weight_v` instead of `*.weight` for the
    /// encoder convolutions. Each nested [`WaveNet::key_remap`] fuses them
    /// into plain weights.
    ///
    /// The `input_conv` / `proj` convolutions are loaded here as well
    /// (plain `weight` / `bias`, with `weight_g` / `weight_v` fusion as a
    /// fallback) instead of relying on the follow-up `load_from` call.
    /// This is required because the Python `ModuleList` interleaves each
    /// coupling layer with a parameter-free flip
    /// (`[layer, flip] * flows`), so the raw checkpoint only carries the
    /// even indices (`flows.0`, `flows.2`, ...). Depending on the store's
    /// `map_indices_contiguous` setting these appear either renumbered
    /// (`flows.{i}`) or raw (`flows.{2*i}`); the cache state is detected
    /// via `flows.1` (a flip, hence parameter-free, under raw keys) and
    /// applied uniformly. The returned store is marked `allow_partial(true)`
    /// for the follow-up `load_from` call.
    ///
    /// # Args
    /// - `store`: Store over the PyTorch checkpoint holding the weights.
    /// - `parents`: Parent key segments of this module in the checkpoint;
    ///   empty when the keys are top-level (`flows.{i}.input_conv.weight`, ...).
    pub fn key_remap(&mut self, store: PytorchStore, parents: &Vec<&str>) -> PytorchStore {
        let mut store = store;
        let prefix = parents.join(".");
        for (i, flow) in self.flows.iter_mut().enumerate() {
            // Burn layer `i` lives at Python ModuleList position `2 * i`.
            let seg = Self::resolve_segment(&mut store, &prefix, i);
            let seg_prefix = if prefix.is_empty() {
                seg.clone()
            } else {
                format!("{prefix}.{seg}")
            };
            Self::load_plain_conv(
                &mut store,
                &mut flow.layer.input_conv,
                &format!("{seg_prefix}.input_conv"),
            );
            Self::load_plain_conv(
                &mut store,
                &mut flow.layer.proj,
                &format!("{seg_prefix}.proj"),
            );

            let mut enc_parents = parents.clone();
            let target = format!("{seg}.encoder");
            enc_parents.push(&target);
            store = flow.layer.encoder.key_remap(store, &enc_parents);
        }
        store.allow_partial(true)
    }

    /// Resolve the checkpoint segment for Burn layer `i`.
    ///
    /// The Python `ModuleList` interleaves each coupling layer with a
    /// parameter-free flip, so raw checkpoints only carry the even
    /// positions (`flows.0`, `flows.2`, ...). The store either preserves
    /// them (`map_indices_contiguous(false)`, required e.g. by DDS
    /// convolutions sharing the store) or renumbers them to be contiguous
    /// (`flows.{i}`). The cache state is detected once via `flows.1`:
    /// under raw keys that position is a flip and carries no parameters,
    /// while under renumbered keys it holds layer 1. Probing per-layer
    /// spellings would be ambiguous (e.g. raw `flows.2` collides with the
    /// mapped spelling of Burn layer 2), so the state decides for all
    /// layers uniformly.
    fn resolve_segment(store: &mut PytorchStore, prefix: &str, i: usize) -> String {
        let key = |seg: &str, leaf: &str| {
            if prefix.is_empty() {
                format!("{seg}.{leaf}")
            } else {
                format!("{prefix}.{seg}.{leaf}")
            }
        };
        let mapped = store
            .get_snapshot(&key("flows.1", "input_conv.weight"))
            .ok()
            .flatten()
            .is_some()
            || store
                .get_snapshot(&key("flows.1", "input_conv.weight_g"))
                .ok()
                .flatten()
                .is_some();
        if mapped {
            format!("flows.{i}")
        } else {
            format!("flows.{}", 2 * i)
        }
    }

    /// Load a plain `Conv1d` (`weight` / `bias`, with `weight_g` /
    /// `weight_v` fusion as fallback) by direct assignment. Missing keys
    /// are silently skipped (covered by `allow_partial` in `load_from`).
    fn load_plain_conv(store: &mut PytorchStore, conv: &mut Conv1d<B>, key_base: &str) {
        let device = conv.weight.device();
        let data_of = |store: &mut PytorchStore, key: &str| -> Option<burn::tensor::TensorData> {
            match store.get_snapshot(key) {
                Ok(Some(snap)) => Some(
                    snap.to_data()
                        .unwrap_or_else(|e| panic!("{key} の取得に失敗しました:\n{e}")),
                ),
                _ => None,
            }
        };
        // Prefer the weight-norm pair; fall back to a plain fused weight.
        let v_data = data_of(store, &format!("{key_base}.weight_v"));
        let g_data = data_of(store, &format!("{key_base}.weight_g"));
        match (g_data, v_data) {
            (Some(g_data), Some(v_data)) => {
                let g: Tensor<B, 3> = Tensor::from_data(g_data, &device);
                let v: Tensor<B, 3> = Tensor::from_data(v_data, &device);
                let norm = v
                    .clone()
                    .powf_scalar(2.0)
                    .sum_dims(&[1, 2])
                    .sqrt()
                    .clamp_min(1e-12);
                conv.weight = Param::from_tensor(v / norm * g);
            }
            _ => {
                if let Some(w_data) = data_of(store, &format!("{key_base}.weight")) {
                    conv.weight = Param::from_tensor(Tensor::from_data(w_data, &device));
                }
            }
        }
        if let Some(b_data) = data_of(store, &format!("{key_base}.bias")) {
            conv.bias = Some(Param::from_tensor(Tensor::from_data(b_data, &device)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::Flex;
    use burn::backend::wgpu::Wgpu;
    use burn_store::{pytorch::PytorchReader, ModuleSnapshot, PytorchStore};

    const REL_TOL: f32 = 0.05;
    const FIXTURE_DIR: &str = "tests/fixtures/modules/residual_coupling";
    type ForwardOutput<B> = Tensor<B, 3>;
    type TestModule<B> = ResidualAffineCouplingBlock<B>;

    fn init_module<B: Backend>(device: &B::Device) -> TestModule<B> {
        ResidualAffineCouplingBlockConfig::new(192, 192, 4)
            .with_kernel_size(5)
            .with_base_dilation(1)
            .with_layers(4)
            .with_global_channels(-1)
            .with_dropout_rate(0.0)
            .with_bias(true)
            .with_use_only_mean(true)
            .init(device)
            .unwrap_or_else(|e| panic!("moduleのinitに失敗.\n{e}"))
    }

    fn key_remap<B: Backend>(store: PytorchStore, module: &mut TestModule<B>) -> PytorchStore {
        module.key_remap(store, &vec![])
    }

    fn forward<B: Backend>(
        module: &TestModule<B>,
        input_reader: &PytorchReader,
        device: &B::Device,
    ) -> ForwardOutput<B> {
        let x = get_tensor(input_reader, "x", device);
        let x_mask = get_tensor(input_reader, "x_mask", device);
        let inverse = get_scalar::<B>(input_reader, "inverse", device) != 0.0;
        module.forward(x, x_mask, None, inverse)
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

    fn to_vec_int<B: Backend, const D: usize>(t: Tensor<B, D, Int>) -> Vec<f32> {
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

    /// Intテンソルをスカラーとして読み出すヘルパー。
    fn get_scalar<B: Backend>(reader: &PytorchReader, key: &str, device: &B::Device) -> f32 {
        let data = reader
            .get(key)
            .unwrap_or_else(|| panic!("{} が見つかりません", key))
            .to_data()
            .unwrap_or_else(|e| panic!("テンソルデータの取得に失敗しました:\n{e}"));
        let t = Tensor::<B, 1, Int>::from_data(data, device);
        to_vec_int(t)[0]
    }
}
