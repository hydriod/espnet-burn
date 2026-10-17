//! WaveNet with global conditioning.
//!
//! Ported from
//! `espnet2/gan_tts/wavenet/wavenet.py` (and `residual_block.py`).
//! Original copyright 2021 Tomoki Hayashi, Apache 2.0.
//! This code is modified from <https://github.com/kan-bayashi/ParallelWaveGAN>.
//!
//! Notes on the port:
//! - `torch.nn.ModuleList` / `Sequential` are plain `Vec`s / structs.
//! - `use_weight_norm` is accepted for API compatibility but is a no-op:
//!   Burn has no weight-norm reparameterization. [`WaveNet::apply_weight_norm`]
//!   and [`WaveNet::remove_weight_norm`] are kept as no-ops.
//! - `aux_channels <= 0` / `global_channels <= 0` (Python default `-1`)
//!   disable the corresponding conditioning branch, matching the Python
//!   `if aux_channels > 0` / `if global_channels > 0` checks.

use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Dropout, DropoutConfig, PaddingConfig1d};
use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid, tanh};
use burn::tensor::TensorData;
use burn_store::{ModuleStore, PytorchStore, PytorchStoreError};

use crate::error::Error;

// ---------------------------------------------------------------------------
// Residual@ block
// ---------------------------------------------------------------------------

/// Configuration for [`WaveNetResidualBlock`].
///
/// Mirrors `ResidualBlock.__init__` in `residual_block.py`.
#[derive(Clone, Debug)]
pub struct WaveNetResidualBlockConfig {
    /// Kernel size of the dilated convolution. Must be odd.
    pub kernel_size: usize,
    /// Number of channels for the residual connection.
    pub residual_channels: usize,
    /// Number of channels of the gated (dilated) convolution output.
    /// Must be even; it is split into two halves for the gated activation.
    pub gate_channels: usize,
    /// Number of channels for the skip connection.
    pub skip_channels: usize,
    /// Number of local-conditioning channels. `<= 0` disables it.
    pub aux_channels: isize,
    /// Number of global-conditioning channels. `<= 0` disables it.
    pub global_channels: isize,
    /// Dropout probability. `0.0` means no dropout.
    pub dropout_rate: f64,
    /// Dilation factor of the dilated convolution.
    pub dilation: usize,
    /// Whether to use bias in convolution layers.
    pub bias: bool,
    /// Whether to scale the residual output by `sqrt(0.5)`.
    pub scale_residual: bool,
}

impl Default for WaveNetResidualBlockConfig {
    fn default() -> Self {
        Self {
            kernel_size: 3,
            residual_channels: 64,
            gate_channels: 128,
            skip_channels: 64,
            aux_channels: 80,
            global_channels: -1,
            dropout_rate: 0.0,
            dilation: 1,
            bias: true,
            scale_residual: false,
        }
    }
}

impl WaveNetResidualBlockConfig {
    pub fn new(
        kernel_size: usize,
        residual_channels: usize,
        gate_channels: usize,
        skip_channels: usize,
    ) -> Self {
        Self {
            kernel_size,
            residual_channels,
            gate_channels,
            skip_channels,
            ..Default::default()
        }
    }

    pub fn with_aux_channels(mut self, aux_channels: isize) -> Self {
        self.aux_channels = aux_channels;
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

    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_scale_residual(mut self, scale_residual: bool) -> Self {
        self.scale_residual = scale_residual;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<WaveNetResidualBlock<B>, Error> {
        if (self.kernel_size - 1) % 2 != 0 {
            return Err(Error::Argument(
                "Not support even number kernel size.".to_string(),
            ));
        }
        if self.gate_channels % 2 != 0 {
            return Err(Error::Argument(
                "gate_channels must be even (it is split for gated activation).".to_string(),
            ));
        }

        // Same-length output: padding = (kernel - 1) / 2 * dilation.
        let padding = (self.kernel_size - 1) / 2 * self.dilation;
        let conv = Conv1dConfig::new(self.residual_channels, self.gate_channels, self.kernel_size)
            .with_padding(PaddingConfig1d::Explicit(padding, padding))
            .with_dilation(self.dilation)
            .with_bias(self.bias)
            .init(device);

        let conv1x1_aux = if self.aux_channels > 0 {
            Some(
                Conv1dConfig::new(self.aux_channels as usize, self.gate_channels, 1)
                    .with_bias(false)
                    .init(device),
            )
        } else {
            None
        };

        let conv1x1_glo = if self.global_channels > 0 {
            Some(
                Conv1dConfig::new(self.global_channels as usize, self.gate_channels, 1)
                    .with_bias(false)
                    .init(device),
            )
        } else {
            None
        };

        // NOTE(kan-bayashi): concat two convs into a single conv for efficiency
        // (integrate res 1x1 + skip 1x1 convs).
        let gate_out_channels = self.gate_channels / 2;
        let conv1x1_out = Conv1dConfig::new(
            gate_out_channels,
            self.residual_channels + self.skip_channels,
            1,
        )
        .with_bias(self.bias)
        .init(device);

        Ok(WaveNetResidualBlock {
            dropout: DropoutConfig::new(self.dropout_rate).init(),
            conv,
            conv1x1_aux,
            conv1x1_glo,
            conv1x1_out,
            residual_channels: self.residual_channels,
            skip_channels: self.skip_channels,
            scale_residual: self.scale_residual,
        })
    }
}

/// Residual block module in WaveNet.
///
/// Corresponds to `ResidualBlock` in `residual_block.py`.
#[derive(Module, Debug)]
pub struct WaveNetResidualBlock<B: Backend> {
    dropout: Dropout,
    conv: Conv1d<B>,
    conv1x1_aux: Option<Conv1d<B>>,
    conv1x1_glo: Option<Conv1d<B>>,
    conv1x1_out: Conv1d<B>,
    residual_channels: usize,
    skip_channels: usize,
    scale_residual: bool,
}

impl<B: Backend> WaveNetResidualBlock<B> {
    /// Calculate forward propagation.
    ///
    /// # Args
    /// - `x`: Input tensor (B, residual_channels, T).
    /// - `x_mask`: Mask tensor (B, 1, T). Multiplied after the output conv.
    /// - `c`: Local conditioning tensor (B, aux_channels, T).
    /// - `g`: Global conditioning tensor (B, global_channels, 1).
    ///
    /// # Returns
    /// - Output tensor for residual connection (B, residual_channels, T).
    /// - Output tensor for skip connection (B, skip_channels, T).
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Option<Tensor<B, 3>>,
        c: Option<Tensor<B, 3>>,
        g: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let residual = x.clone();
        let x = self.dropout.forward(x);
        let x = self.conv.forward(x);

        // Split into two parts for gated activation.
        let parts = x.chunk(2, 1);
        let (mut xa, mut xb) = (parts[0].clone(), parts[1].clone());

        // Local conditioning.
        if let (Some(aux), Some(cond)) = (&self.conv1x1_aux, c) {
            let cond = aux.forward(cond);
            let cond_parts = cond.chunk(2, 1);
            xa = xa + cond_parts[0].clone();
            xb = xb + cond_parts[1].clone();
        }

        // Global conditioning.
        if let (Some(glo), Some(cond)) = (&self.conv1x1_glo, g) {
            let cond = glo.forward(cond);
            let cond_parts = cond.chunk(2, 1);
            xa = xa + cond_parts[0].clone();
            xb = xb + cond_parts[1].clone();
        }

        let mut x = tanh(xa) * sigmoid(xb);

        // Residual + skip 1x1 conv.
        x = self.conv1x1_out.forward(x);
        if let Some(mask) = x_mask {
            x = x * mask;
        }

        // Split integrated conv results.
        let x_res = x.clone().slice_dim(1, 0..self.residual_channels);
        let s = x.slice_dim(
            1,
            self.residual_channels..self.residual_channels + self.skip_channels,
        );

        // For residual connection.
        let mut x = x_res + residual;
        if self.scale_residual {
            x = x * (0.5f64).sqrt();
        }

        (x, s)
    }
}

// ---------------------------------------------------------------------------
// Output (last) conv: ReLU -> Conv1x1 -> ReLU -> Conv1x1
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct LastConvModule<B: Backend> {
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
}

impl<B: Backend> LastConvModule<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = relu(x);
        let x = self.conv1.forward(x);
        let x = relu(x);
        self.conv2.forward(x)
    }
}

// ---------------------------------------------------------------------------
// WaveNet
// ---------------------------------------------------------------------------

/// Configuration for [`WaveNet`].
///
/// Mirrors `WaveNet.__init__` in `wavenet.py`.
#[derive(Clone, Debug)]
pub struct WaveNetConfig {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Kernel size of dilated convolutions.
    pub kernel_size: usize,
    /// Number of residual block layers.
    pub layers: usize,
    /// Number of stacks, i.e. dilation cycles. Must divide `layers`.
    pub stacks: usize,
    /// Base dilation factor.
    pub base_dilation: usize,
    /// Number of channels in residual conv.
    pub residual_channels: usize,
    /// Number of channels for local conditioning. `<= 0` disables it.
    pub aux_channels: isize,
    /// Number of channels in gated conv. Must be even.
    pub gate_channels: usize,
    /// Number of channels in skip conv.
    pub skip_channels: usize,
    /// Number of channels for global conditioning. `<= 0` disables it.
    pub global_channels: isize,
    /// Dropout rate. `0.0` means no dropout applied.
    pub dropout_rate: f64,
    /// Whether to use bias in conv layers.
    pub bias: bool,
    /// Accepted for API compatibility; no-op in this Burn port.
    pub use_weight_norm: bool,
    /// Whether to use the first 1x1 conv layer.
    pub use_first_conv: bool,
    /// Whether to use the last conv layers.
    pub use_last_conv: bool,
    /// Whether to scale the residual outputs.
    pub scale_residual: bool,
    /// Whether to scale the skip connection outputs.
    pub scale_skip_connect: bool,
}

impl Default for WaveNetConfig {
    fn default() -> Self {
        Self {
            in_channels: 1,
            out_channels: 1,
            kernel_size: 3,
            layers: 30,
            stacks: 3,
            base_dilation: 2,
            residual_channels: 64,
            aux_channels: -1,
            gate_channels: 128,
            skip_channels: 64,
            global_channels: -1,
            dropout_rate: 0.0,
            bias: true,
            use_weight_norm: true,
            use_first_conv: false,
            use_last_conv: false,
            scale_residual: false,
            scale_skip_connect: false,
        }
    }
}

impl WaveNetConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_in_channels(mut self, in_channels: usize) -> Self {
        self.in_channels = in_channels;
        self
    }

    pub fn with_out_channels(mut self, out_channels: usize) -> Self {
        self.out_channels = out_channels;
        self
    }

    pub fn with_kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = kernel_size;
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

    pub fn with_base_dilation(mut self, base_dilation: usize) -> Self {
        self.base_dilation = base_dilation;
        self
    }

    pub fn with_residual_channels(mut self, residual_channels: usize) -> Self {
        self.residual_channels = residual_channels;
        self
    }

    pub fn with_aux_channels(mut self, aux_channels: isize) -> Self {
        self.aux_channels = aux_channels;
        self
    }

    pub fn with_gate_channels(mut self, gate_channels: usize) -> Self {
        self.gate_channels = gate_channels;
        self
    }

    pub fn with_skip_channels(mut self, skip_channels: usize) -> Self {
        self.skip_channels = skip_channels;
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

    pub fn with_use_first_conv(mut self, use_first_conv: bool) -> Self {
        self.use_first_conv = use_first_conv;
        self
    }

    pub fn with_use_last_conv(mut self, use_last_conv: bool) -> Self {
        self.use_last_conv = use_last_conv;
        self
    }

    pub fn with_scale_residual(mut self, scale_residual: bool) -> Self {
        self.scale_residual = scale_residual;
        self
    }

    pub fn with_scale_skip_connect(mut self, scale_skip_connect: bool) -> Self {
        self.scale_skip_connect = scale_skip_connect;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<WaveNet<B>, Error> {
        if self.stacks == 0 || self.layers % self.stacks != 0 {
            return Err(Error::Argument(format!(
                "layers ({}) must be divisible by stacks ({}).",
                self.layers, self.stacks
            )));
        }
        let layers_per_stack = self.layers / self.stacks;

        let first_conv = if self.use_first_conv {
            Some(
                Conv1dConfig::new(self.in_channels, self.residual_channels, 1)
                    .with_bias(true)
                    .init(device),
            )
        } else {
            None
        };

        let mut conv_layers = Vec::with_capacity(self.layers);
        for layer in 0..self.layers {
            let dilation = self.base_dilation.pow((layer % layers_per_stack) as u32);
            conv_layers.push(
                WaveNetResidualBlockConfig {
                    kernel_size: self.kernel_size,
                    residual_channels: self.residual_channels,
                    gate_channels: self.gate_channels,
                    skip_channels: self.skip_channels,
                    aux_channels: self.aux_channels,
                    global_channels: self.global_channels,
                    dropout_rate: self.dropout_rate,
                    dilation,
                    bias: self.bias,
                    scale_residual: self.scale_residual,
                }
                .init(device)?,
            );
        }

        let last_conv = if self.use_last_conv {
            Some(LastConvModule {
                conv1: Conv1dConfig::new(self.skip_channels, self.skip_channels, 1)
                    .with_bias(true)
                    .init(device),
                conv2: Conv1dConfig::new(self.skip_channels, self.out_channels, 1)
                    .with_bias(true)
                    .init(device),
            })
        } else {
            None
        };

        Ok(WaveNet {
            first_conv,
            conv_layers,
            last_conv,
            layers: self.layers,
            stacks: self.stacks,
            kernel_size: self.kernel_size,
            base_dilation: self.base_dilation,
            use_first_conv: self.use_first_conv,
            use_last_conv: self.use_last_conv,
            scale_skip_connect: self.scale_skip_connect,
        })
    }
}

/// WaveNet with global conditioning.
///
/// Corresponds to `WaveNet` in `wavenet.py`.
#[derive(Module, Debug)]
pub struct WaveNet<B: Backend> {
    first_conv: Option<Conv1d<B>>,
    conv_layers: Vec<WaveNetResidualBlock<B>>,
    last_conv: Option<LastConvModule<B>>,
    layers: usize,
    stacks: usize,
    kernel_size: usize,
    base_dilation: usize,
    use_first_conv: bool,
    use_last_conv: bool,
    scale_skip_connect: bool,
}

impl<B: Backend> WaveNet<B> {
    /// Calculate forward propagation.
    ///
    /// # Args
    /// - `x`: Input signal (B, 1, T) if `use_first_conv`, else
    ///   (B, residual_channels, T).
    /// - `x_mask`: Mask tensor (B, 1, T).
    /// - `c`: Local conditioning features (B, aux_channels, T).
    /// - `g`: Global conditioning features (B, global_channels, 1).
    ///
    /// # Returns
    /// - Output tensor (B, out_channels, T) if `use_last_conv`, else
    ///   (B, residual_channels, T) — more precisely (B, skip_channels, T)
    ///   after skip accumulation, matching the Python implementation where
    ///   the residual loop output `x` is replaced by summed skips.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Option<Tensor<B, 3>>,
        c: Option<Tensor<B, 3>>,
        g: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        // Encode to hidden representation.
        let mut x = match (&self.first_conv, self.use_first_conv) {
            (Some(first_conv), true) => first_conv.forward(x),
            _ => x,
        };

        // Residual blocks; accumulate skip connections.
        let mut skips: Option<Tensor<B, 3>> = None;
        for f in &self.conv_layers {
            let (x_next, h) = f.forward(x, x_mask.clone(), c.clone(), g.clone());
            x = x_next;
            skips = Some(match skips {
                None => h,
                Some(s) => s + h,
            });
        }
        // `conv_layers` is always non-empty when constructed via config
        // (Python defaults to 30 layers), so `skips` is `Some` here.
        let mut x = skips.unwrap_or(x);
        if self.scale_skip_connect {
            x = x * (1.0 / self.conv_layers.len() as f64).sqrt();
        }

        // Apply final layers.
        match (&self.last_conv, self.use_last_conv) {
            (Some(last_conv), true) => last_conv.forward(x),
            _ => x,
        }
    }

    /// Reconstruct plain convolution weights from weight-norm parameters.
    ///
    /// Checkpoints saved with `use_weight_norm=True` (the ESPnet default)
    /// store `*.weight_g` and `*.weight_v` instead of `*.weight`.
    /// Burn's `Conv1d` has no weight-norm reparameterization, so the fused
    /// weight `w = g * v / ||v||` (norm over all dims except dim 0) is
    /// computed here and assigned to `*.weight`, matching
    /// `torch.nn.utils.weight_norm` semantics.
    ///
    /// Biases (`*.bias`) are assigned here as well (not left to the
    /// follow-up `load_from` call): `load_from` matches exact module paths
    /// and therefore cannot see parameters when this remap runs under a
    /// parent prefix (e.g. inside the VITS generator). Layers without
    /// matching keys in `store` are left untouched, so the returned store
    /// is marked `allow_partial(true)` for the follow-up `load_from` call.
    ///
    /// # Args
    /// - `reader`: Reader over the PyTorch checkpoint holding the
    ///   `weight_g` / `weight_v` tensors.
    /// - `device`: Device used for the intermediate fused tensors.
    /// - `key_prefix`: Prefix of the WaveNet keys in the checkpoint
    ///   (e.g. `"decoder"`); empty when the keys are top-level
    ///   (`conv_layers.{i}.conv.weight_g`, ...).
    pub fn key_remap(&mut self, store: PytorchStore, parents: &Vec<&str>) -> PytorchStore {
        let mut store = store;
        let mut parents = parents.clone();
        parents.push("");
        let base = parents.join(".");
        for (i, block) in self.conv_layers.iter_mut().enumerate() {
            Self::fuse_conv_weight(
                &mut store,
                &mut block.conv,
                &format!("{base}conv_layers.{i}.conv"),
            )
            .unwrap_or_else(|e| {
                println!("conv_layers.{i}.convの重みをロードできませんでした。\n{e}")
            });
            Self::fuse_conv_weight(
                &mut store,
                &mut block.conv1x1_out,
                &format!("{base}conv_layers.{i}.conv1x1_out"),
            )
            .unwrap_or_else(|e| {
                println!("conv_layers.{i}.conv1x1の重みをロードできませんでした。\n{e}")
            });
            if let Some(aux) = block.conv1x1_aux.as_mut() {
                Self::fuse_conv_weight(
                    &mut store,
                    aux,
                    &format!("{base}conv_layers.{i}.conv1x1_aux"),
                )
                .unwrap_or_else(|e| {
                    println!("conv_layers.{i}.conv1x1_auxの重みをロードできませんでした。\n{e}")
                });
            }
            if let Some(glo) = block.conv1x1_glo.as_mut() {
                Self::fuse_conv_weight(
                    &mut store,
                    glo,
                    &format!("{base}conv_layers.{i}.conv1x1_glo"),
                )
                .unwrap_or_else(|e| {
                    println!("conv_layers.{i}.conv1x1_gloの重みをロードできませんでした。\n{e}")
                });
            }
        }
        store.allow_partial(true)
    }

    fn fuse_conv_weight(
        store: &mut PytorchStore,
        conv: &mut Conv1d<B>,
        key_base: &str,
    ) -> Result<(), PytorchStoreError> {
        let device = &conv.weight.device();

        let data_of = |store: &mut PytorchStore, key: &str| -> Option<TensorData> {
            match store.get_snapshot(key) {
                Ok(Some(snap)) => Some(
                    snap.to_data()
                        .unwrap_or_else(|e| panic!("{key} の取得に失敗しました:\n{e}")),
                ),
                _ => None,
            }
        };

        // Prefer the weight-norm pair; fall back to a plain fused weight
        // (checkpoints saved after `remove_weight_norm()`).
        let v_data = data_of(store, &format!("{key_base}.weight_v"));
        let g_data = data_of(store, &format!("{key_base}.weight_g"));
        match (g_data, v_data) {
            (Some(g_data), Some(v_data)) => {
                let g: Tensor<B, 3> = Tensor::from_data(g_data, device);
                let v: Tensor<B, 3> = Tensor::from_data(v_data, device);
                // norm over all dims except dim 0, keep dims -> [C, 1, 1].
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
                    conv.weight = Param::from_tensor(Tensor::from_data(w_data, device));
                }
            }
        }
        if let Some(b_data) = data_of(store, &format!("{key_base}.bias")) {
            conv.bias = Some(Param::from_tensor(Tensor::from_data(b_data, device)));
        }
        Ok(())
    }

    /// Compute the receptive field size from hyperparameters.
    pub fn get_receptive_field_size(
        layers: usize,
        stacks: usize,
        kernel_size: usize,
        base_dilation: usize,
    ) -> usize {
        assert!(
            stacks != 0 && layers % stacks == 0,
            "layers must be divisible by stacks."
        );
        let layers_per_cycle = layers / stacks;
        let dilations: Vec<usize> = (0..layers)
            .map(|i| base_dilation.pow((i % layers_per_cycle) as u32))
            .collect();
        (kernel_size - 1) * dilations.iter().sum::<usize>() + 1
    }

    /// Return receptive field size.
    pub fn receptive_field_size(&self) -> usize {
        Self::get_receptive_field_size(
            self.layers,
            self.stacks,
            self.kernel_size,
            self.base_dilation,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::Flex;
    use burn::backend::wgpu::Wgpu;
    use burn_store::{pytorch::PytorchReader, ModuleSnapshot, PytorchStore};

    const REL_TOL: f32 = 0.05;
    const FIXTURE_DIR: &str = "tests/fixtures/modules/wavenet";
    type ForwardOutput<B> = Tensor<B, 3>;
    type TestModule<B> = WaveNet<B>;

    fn init_module<B: Backend>(device: &B::Device) -> TestModule<B> {
        WaveNetConfig::new()
            .with_in_channels(0)
            .with_out_channels(0)
            .with_kernel_size(5)
            .with_layers(4)
            .with_stacks(1)
            .with_base_dilation(1)
            .with_residual_channels(192)
            .with_gate_channels(384)
            .with_skip_channels(192)
            .with_global_channels(-1)
            .with_dropout_rate(0.0)
            .with_bias(true)
            .with_scale_skip_connect(true)
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
        // forward内で生成される乱数の影響を消すためnoise_scaleを0.0に設定
        module.forward(
            get_tensor(input_reader, "x", device),
            Some(get_tensor(input_reader, "x_mask", device)),
            None,
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
        let mut weight_store = key_remap(weight_store, &mut module);
        module.load_from(&mut weight_store).unwrap_or_else(|e| {
            panic!("重みのロードに失敗しました。key_remapやshapeの不一致を確認してください:\n{e}")
        });
        // weight_norm 由来の `weight_g` / `weight_v` を融合して `weight` に設定する。
        /*
        let weight_reader = PytorchReader::new(&weight_path)
            .unwrap_or_else(|e| panic!("weight.pth の読み込みに失敗しました:\n{e}"));
        module.key_remap(&weight_reader, &device, "");
        */

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
