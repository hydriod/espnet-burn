//! HiFi-GAN generator.
//!
//! Ported from `espnet2/gan_tts/hifigan/hifigan.py`
//! (`HiFiGANGenerator`) and `espnet2/gan_tts/hifigan/residual_block.py`
//! (`ResidualBlock`).
//! Original copyright 2021 Tomoki Hayashi, Apache 2.0.
//! This code is modified from <https://github.com/kan-bayashi/ParallelWaveGAN>.
//!
//! Notes on the port:
//! - `torch.nn.ModuleList` / `Sequential` are plain `Vec`s. The stateless
//!   activations (`LeakyReLU`, `Tanh`) are applied functionally, so they are
//!   not stored as fields.
//! - Activation slopes follow the Python defaults: `negative_slope=0.1` for
//!   the upsample and residual-block stages
//!   (`nonlinear_activation_params`), and the default slope (`0.01`) for the
//!   `LeakyReLU` inside `output_conv` (plain `torch.nn.LeakyReLU()`).
//! - `global_channels == 0` disables the global-conditioning branch,
//!   matching the Python `if global_channels > 0` check (Python default
//!   is `-1`; this config uses `usize` so `0` means "none").
//! - `use_weight_norm` is accepted for API compatibility but is a no-op:
//!   Burn has no weight-norm reparameterization. Checkpoints saved with
//!   `use_weight_norm=True` (the ESPnet default) store `*.weight_g` and
//!   `*.weight_v` instead of `*.weight`; use [`HiFiGANGenerator::key_remap`]
//!   to fuse them into plain `weight` tensors on load (equivalent to
//!   `remove_weight_norm()` for inference).

use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use burn::nn::PaddingConfig1d;
use burn::prelude::*;
use burn::tensor::activation::{leaky_relu, tanh};
use burn_store::{ModuleStore, PytorchStore, PytorchStoreError, TensorSnapshot};

use crate::error::Error;

/// LeakyReLU slope for the upsample / residual-block stages.
const NEGATIVE_SLOPE: f64 = 0.1;
/// LeakyReLU slope inside `output_conv` (plain `torch.nn.LeakyReLU()`).
const OUTPUT_NEGATIVE_SLOPE: f64 = 0.01;

// ---------------------------------------------------------------------------
// Residual block
// ---------------------------------------------------------------------------

/// Configuration for [`HiFiGANResidualBlock`].
///
/// Mirrors `ResidualBlock.__init__` in `residual_block.py`.
#[derive(Clone, Debug)]
pub struct HiFiGANResidualBlockConfig {
    /// Kernel size of the dilated convolutions. Must be odd.
    pub kernel_size: usize,
    /// Number of channels for the convolution layers.
    pub channels: usize,
    /// List of dilation factors (one `convs1` / `convs2` pair each).
    pub dilations: Vec<usize>,
    /// Whether to use bias in convolution layers.
    pub bias: bool,
    /// Whether to use the additional (`convs2`) convolution layers.
    pub use_additional_convs: bool,
}

impl HiFiGANResidualBlockConfig {
    pub fn new(kernel_size: usize, channels: usize, dilations: Vec<usize>) -> Self {
        Self {
            kernel_size,
            channels,
            dilations,
            bias: true,
            use_additional_convs: true,
        }
    }

    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_use_additional_convs(mut self, use_additional_convs: bool) -> Self {
        self.use_additional_convs = use_additional_convs;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<HiFiGANResidualBlock<B>, Error> {
        if (self.kernel_size - 1) % 2 != 0 {
            return Err(Error::Argument(
                "Kernel size must be odd number.".to_string(),
            ));
        }
        let mut convs1 = Vec::with_capacity(self.dilations.len());
        let mut convs2 = Vec::with_capacity(self.dilations.len());
        for &dilation in &self.dilations {
            // Same-length output: padding = (kernel - 1) / 2 * dilation.
            let padding = (self.kernel_size - 1) / 2 * dilation;
            convs1.push(
                Conv1dConfig::new(self.channels, self.channels, self.kernel_size)
                    .with_padding(PaddingConfig1d::Explicit(padding, padding))
                    .with_dilation(dilation)
                    .with_bias(self.bias)
                    .init(device),
            );
            if self.use_additional_convs {
                let padding = (self.kernel_size - 1) / 2;
                convs2.push(
                    Conv1dConfig::new(self.channels, self.channels, self.kernel_size)
                        .with_padding(PaddingConfig1d::Explicit(padding, padding))
                        .with_bias(self.bias)
                        .init(device),
                );
            }
        }
        Ok(HiFiGANResidualBlock {
            convs1,
            convs2,
            use_additional_convs: self.use_additional_convs,
        })
    }
}

/// Residual block module in HiFi-GAN.
///
/// Corresponds to `ResidualBlock` in `residual_block.py`.
#[derive(Module, Debug)]
pub struct HiFiGANResidualBlock<B: Backend> {
    convs1: Vec<Conv1d<B>>,
    convs2: Vec<Conv1d<B>>,
    use_additional_convs: bool,
}

impl<B: Backend> HiFiGANResidualBlock<B> {
    /// Calculate forward propagation.
    ///
    /// # Args
    /// - `x`: Input tensor (B, channels, T).
    ///
    /// # Returns
    /// - Output tensor (B, channels, T).
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = x;
        for (i, conv) in self.convs1.iter().enumerate() {
            let mut xt = leaky_relu(x.clone(), NEGATIVE_SLOPE);
            xt = conv.forward(xt);
            if self.use_additional_convs {
                xt = leaky_relu(xt, NEGATIVE_SLOPE);
                xt = self.convs2[i].forward(xt);
            }
            x = xt + x;
        }
        x
    }
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

/// - `in_channels` : Number of input channels.
/// - `out_channels` : Number of output channels.
/// - `channels` : Number of hidden representation channels.
/// - `global_channels` : Number of global conditioning channels.
/// - `kernel_size` : Kernel size of initial and final conv layer.
/// - `upsample_scales` : List of upsampling scales.
/// - `upsample_kernel_sizes` : List of kernel sizes for upsample layers.
/// - `resblock_kernel_sizes` : List of kernel sizes for residual blocks.
/// - `resblock_dilations` : List of list of dilations for residual
///     blocks.
/// - `use_additional_convs` : Whether to use additional conv layers in
///     residual blocks.
/// - `bias` : Whether to add bias parameter in convolution layers.
#[derive(Clone, Debug)]
pub struct HiFiGANGeneratorConfig {
    in_channels: usize,
    out_channels: usize,
    channels: usize,
    global_channels: usize,
    kernel_size: usize,
    upsample_scales: Vec<usize>,
    upsample_kernel_sizes: Vec<usize>,
    resblock_kernel_sizes: Vec<usize>,
    resblock_dilations: Vec<Vec<usize>>,
    use_additional_convs: bool,
    bias: bool,
}

impl HiFiGANGeneratorConfig {
    /// Create a new [`HiFiGANGeneratorConfig`].
    ///
    /// Mirrors the `HiFiGANGenerator(...)` construction in
    /// `espnet2/gan_tts/vits/generator.py` (`in_channels=hidden_channels`,
    /// `out_channels=1`). `use_additional_convs` and `bias` follow the
    /// Python defaults (`True`).
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        channels: usize,
        global_channels: usize,
        kernel_size: usize,
        upsample_scales: Vec<usize>,
        upsample_kernel_sizes: Vec<usize>,
        resblock_kernel_sizes: Vec<usize>,
        resblock_dilations: Vec<Vec<usize>>,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            channels,
            global_channels,
            kernel_size,
            upsample_scales,
            upsample_kernel_sizes,
            resblock_kernel_sizes,
            resblock_dilations,
            use_additional_convs: true,
            bias: true,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<HiFiGANGenerator<B>, Error> {
        if (self.kernel_size - 1) % 2 != 0 {
            return Err(Error::Argument(
                "Kernel size must be odd number.".to_string(),
            ));
        }
        if self.upsample_scales.len() != self.upsample_kernel_sizes.len() {
            return Err(Error::Argument(format!(
                "upsample_scales (len {}) and upsample_kernel_sizes (len {}) must have the same length.",
                self.upsample_scales.len(),
                self.upsample_kernel_sizes.len()
            )));
        }
        if self.resblock_dilations.len() != self.resblock_kernel_sizes.len() {
            return Err(Error::Argument(format!(
                "resblock_dilations (len {}) and resblock_kernel_sizes (len {}) must have the same length.",
                self.resblock_dilations.len(),
                self.resblock_kernel_sizes.len()
            )));
        }

        let num_upsamples = self.upsample_kernel_sizes.len();
        let num_blocks = self.resblock_kernel_sizes.len();

        let padding = (self.kernel_size - 1) / 2;
        let input_conv = Conv1dConfig::new(self.in_channels, self.channels, self.kernel_size)
            .with_padding(PaddingConfig1d::Explicit(padding, padding))
            .with_bias(self.bias)
            .init(device);

        let mut upsamples = Vec::with_capacity(num_upsamples);
        let mut blocks = Vec::with_capacity(num_upsamples * num_blocks);
        for i in 0..num_upsamples {
            let scale = self.upsample_scales[i];
            let kernel = self.upsample_kernel_sizes[i];
            if kernel != 2 * scale {
                return Err(Error::Argument(format!(
                    "upsample_kernel_sizes[{i}] ({kernel}) must be 2 * upsample_scales[{i}] ({scale})."
                )));
            }
            upsamples.push(
                ConvTranspose1dConfig::new([self.channels >> i, self.channels >> (i + 1)], kernel)
                    .with_stride(scale)
                    .with_padding(scale / 2 + scale % 2)
                    .with_padding_out(scale % 2)
                    .with_bias(self.bias)
                    .init(device),
            );
            for j in 0..num_blocks {
                blocks.push(
                    HiFiGANResidualBlockConfig::new(
                        self.resblock_kernel_sizes[j],
                        self.channels >> (i + 1),
                        self.resblock_dilations[j].clone(),
                    )
                    .with_bias(self.bias)
                    .with_use_additional_convs(self.use_additional_convs)
                    .init(device)?,
                );
            }
        }

        let output_conv = Conv1dConfig::new(
            self.channels >> num_upsamples,
            self.out_channels,
            self.kernel_size,
        )
        .with_padding(PaddingConfig1d::Explicit(padding, padding))
        .with_bias(self.bias)
        .init(device);

        let global_conv = if self.global_channels > 0 {
            Some(
                Conv1dConfig::new(self.global_channels, self.channels, 1)
                    .with_bias(self.bias)
                    .init(device),
            )
        } else {
            None
        };

        Ok(HiFiGANGenerator {
            input_conv,
            upsamples,
            blocks,
            output_conv,
            global_conv,
            num_upsamples,
            num_blocks,
        })
    }
}

#[derive(Module, Debug)]
pub struct HiFiGANGenerator<B: Backend> {
    input_conv: Conv1d<B>,
    upsamples: Vec<ConvTranspose1d<B>>,
    blocks: Vec<HiFiGANResidualBlock<B>>,
    output_conv: Conv1d<B>,
    global_conv: Option<Conv1d<B>>,
    num_upsamples: usize,
    num_blocks: usize,
}

impl<B: Backend> HiFiGANGenerator<B> {
    /// Calculate forward propagation.
    ///
    /// # Args
    /// - `c`: Input tensor (B, in_channels, T).
    /// - `g`: Global conditioning tensor (B, global_channels, 1).
    ///
    /// # Returns
    /// - Output tensor (B, out_channels, T * upsample_factor).
    pub fn forward(&self, c: Tensor<B, 3>, g: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let mut c = self.input_conv.forward(c);
        if let Some(g) = g {
            match &self.global_conv {
                Some(global_conv) => c = c + global_conv.forward(g),
                None => panic!("global conditioning given but global_channels == 0"),
            }
        }
        for i in 0..self.num_upsamples {
            c = leaky_relu(c, NEGATIVE_SLOPE);
            c = self.upsamples[i].forward(c);
            let mut cs: Option<Tensor<B, 3>> = None;
            for j in 0..self.num_blocks {
                let h = self.blocks[i * self.num_blocks + j].forward(c.clone());
                cs = Some(match cs {
                    None => h,
                    Some(s) => s + h,
                });
            }
            // `num_blocks` is always non-empty when constructed via config.
            c = cs.unwrap() * (1.0 / self.num_blocks as f64);
        }
        c = leaky_relu(c, OUTPUT_NEGATIVE_SLOPE);
        c = self.output_conv.forward(c);
        tanh(c)
    }

    /// Load weights from a PyTorch checkpoint using the Python key layout.
    ///
    /// Checkpoints saved with `use_weight_norm=True` (the ESPnet default)
    /// store `*.weight_g` and `*.weight_v` instead of `*.weight`.
    /// Burn's convolutions have no weight-norm reparameterization, so the
    /// fused weight `w = g * v / ||v||` (norm over all dims except dim 0)
    /// is computed here and assigned to `weight`, matching
    /// `torch.nn.utils.weight_norm` semantics. Biases (`*.bias`) are
    /// loaded as well.
    ///
    /// The Python `Sequential` wrappers contribute an extra index level
    /// in the raw checkpoint keys (e.g. `upsamples.0.1.weight_g`,
    /// `blocks.0.convs1.0.1.bias`, `output_conv.1.weight_v`).
    /// Note that `PytorchStore` renumbers numeric path segments per
    /// parent to be contiguous from 0 (`map_indices_contiguous`), so
    /// these keys are visible here as `upsamples.{i}.0.*`,
    /// `blocks.{b}.convs1.{d}.0.*` / `convs2.{d}.0.*` and
    /// `output_conv.0.*`.
    ///
    /// However, sibling modules (e.g. the VITS duration predictor's DDS
    /// convolutions, whose raw indices must be preserved) may disable
    /// that renumbering on the shared store. To work with either cache
    /// state, each wrapped layer probes both the mapped (`.0`) and the
    /// raw (`.1`) key variants and uses whichever exists.
    ///
    /// Layers without matching keys in `store` are left untouched, so the
    /// returned store is marked `allow_partial(true)` for the follow-up
    /// `load_from` call.
    ///
    /// # Args
    /// - `store`: Store over the PyTorch checkpoint holding the weights.
    /// - `parents`: Parent key segments of this module in the checkpoint
    ///   (e.g. `vec!["tts", "generator", "decoder"]`); empty when the keys
    ///   are top-level (`input_conv.weight_g`, ...).
    pub fn key_remap(&mut self, store: PytorchStore, parents: &Vec<&str>) -> PytorchStore {
        let mut store = store;
        let mut parents = parents.clone();
        parents.push("");
        let base = parents.join(".");

        Self::load_conv1d(
            &mut store,
            &mut self.input_conv,
            &format!("{base}input_conv"),
        );
        if let Some(global_conv) = self.global_conv.as_mut() {
            Self::load_conv1d(&mut store, global_conv, &format!("{base}global_conv"));
        }
        for (i, upsample) in self.upsamples.iter_mut().enumerate() {
            let key_base = Self::resolve_base(
                &mut store,
                &format!("{base}upsamples.{i}.0"),
                &format!("{base}upsamples.{i}.1"),
            );
            Self::load_conv_transpose1d(&mut store, upsample, &key_base);
        }
        for (b, block) in self.blocks.iter_mut().enumerate() {
            for (d, conv) in block.convs1.iter_mut().enumerate() {
                let key_base = Self::resolve_base(
                    &mut store,
                    &format!("{base}blocks.{b}.convs1.{d}.0"),
                    &format!("{base}blocks.{b}.convs1.{d}.1"),
                );
                Self::load_conv1d(&mut store, conv, &key_base);
            }
            for (d, conv) in block.convs2.iter_mut().enumerate() {
                let key_base = Self::resolve_base(
                    &mut store,
                    &format!("{base}blocks.{b}.convs2.{d}.0"),
                    &format!("{base}blocks.{b}.convs2.{d}.1"),
                );
                Self::load_conv1d(&mut store, conv, &key_base);
            }
        }
        let key_base = Self::resolve_base(
            &mut store,
            &format!("{base}output_conv.0"),
            &format!("{base}output_conv.1"),
        );
        Self::load_conv1d(&mut store, &mut self.output_conv, &key_base);
        store.allow_partial(true)
    }

    /// Pick whichever key base exists in the store: the mapped base first
    /// (e.g. `upsamples.0.0`, visible when `map_indices_contiguous` is
    /// enabled), falling back to the raw base (e.g. `upsamples.0.1`).
    fn resolve_base(store: &mut PytorchStore, mapped: &str, raw: &str) -> String {
        for cand in [mapped, raw] {
            for suffix in [".weight", ".weight_g", ".bias"] {
                match store.get_snapshot(&format!("{cand}{suffix}")) {
                    Ok(Some(_)) => return cand.to_string(),
                    _ => {}
                }
            }
        }
        mapped.to_string()
    }

    fn load_conv1d(store: &mut PytorchStore, conv: &mut Conv1d<B>, key_base: &str) {
        let device = conv.weight.device();
        if let Err(e) = Self::load_conv_weight(store, key_base, &device, &mut conv.weight) {
            println!("{key_base}の重みをロードできませんでした。\n{e}");
        }
        match store.get_snapshot(&format!("{key_base}.bias")) {
            Ok(Some(snap)) => {
                if conv.bias.is_some() {
                    let bias: Tensor<B, 1> =
                        Tensor::from_data(snap_to_data(&format!("{key_base}.bias"), snap), &device);
                    conv.bias = Some(Param::from_tensor(bias));
                }
            }
            Ok(None) => {}
            Err(e) => println!("{key_base}.biasのロードに失敗しました。\n{e}"),
        }
    }

    fn load_conv_transpose1d(
        store: &mut PytorchStore,
        conv: &mut ConvTranspose1d<B>,
        key_base: &str,
    ) {
        let device = conv.weight.device();
        if let Err(e) = Self::load_conv_weight(store, key_base, &device, &mut conv.weight) {
            println!("{key_base}の重みをロードできませんでした。\n{e}");
        }
        match store.get_snapshot(&format!("{key_base}.bias")) {
            Ok(Some(snap)) => {
                if conv.bias.is_some() {
                    let bias: Tensor<B, 1> =
                        Tensor::from_data(snap_to_data(&format!("{key_base}.bias"), snap), &device);
                    conv.bias = Some(Param::from_tensor(bias));
                }
            }
            Ok(None) => {}
            Err(e) => println!("{key_base}.biasのロードに失敗しました。\n{e}"),
        }
    }

    fn load_conv_weight(
        store: &mut PytorchStore,
        key_base: &str,
        device: &B::Device,
        weight: &mut Param<Tensor<B, 3>>,
    ) -> Result<(), PytorchStoreError> {
        // Prefer the weight-norm pair; fall back to a plain fused weight
        // (checkpoints saved after `remove_weight_norm()`).
        // NOTE: each snapshot is copied out to an owned `TensorData`
        // immediately because `get_snapshot` borrows `store`.
        let v_data = match store.get_snapshot(&format!("{key_base}.weight_v"))? {
            Some(snap) => Some(snap_to_data(&format!("{key_base}.weight_v"), snap)),
            None => None,
        };
        let g_data = match store.get_snapshot(&format!("{key_base}.weight_g"))? {
            Some(snap) => Some(snap_to_data(&format!("{key_base}.weight_g"), snap)),
            None => None,
        };
        match (g_data, v_data) {
            (Some(g_data), Some(v_data)) => {
                let g: Tensor<B, 3> = Tensor::from_data(g_data, device);
                let v: Tensor<B, 3> = Tensor::from_data(v_data, device);
                // norm over all dims except dim 0, keep dims.
                let norm = v
                    .clone()
                    .powf_scalar(2.0)
                    .sum_dims(&[1, 2])
                    .sqrt()
                    .clamp_min(1e-12);
                *weight = Param::from_tensor(v / norm * g);
            }
            _ => {
                if let Some(w_snap) = store.get_snapshot(&format!("{key_base}.weight"))? {
                    let w: Tensor<B, 3> = Tensor::from_data(
                        snap_to_data(&format!("{key_base}.weight"), w_snap),
                        device,
                    );
                    *weight = Param::from_tensor(w);
                }
            }
        }
        Ok(())
    }
}

fn snap_to_data(key: &str, snap: &TensorSnapshot) -> burn::tensor::TensorData {
    snap.to_data()
        .unwrap_or_else(|e| panic!("{key} の取得に失敗しました:\n{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::Flex;
    use burn::backend::wgpu::Wgpu;
    use burn_store::{pytorch::PytorchReader, ModuleSnapshot, PytorchStore};

    const REL_TOL: f32 = 0.05;
    const FIXTURE_DIR: &str = "tests/fixtures/modules/hifi_gan/generator";
    type ForwardOutput<B> = Tensor<B, 3>;
    type TestModule<B> = HiFiGANGenerator<B>;

    fn init_module<B: Backend>(device: &B::Device) -> TestModule<B> {
        HiFiGANGeneratorConfig {
            in_channels: 192,
            out_channels: 1,
            channels: 512,
            global_channels: 0,
            kernel_size: 7,
            upsample_scales: vec![8, 8, 2, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilations: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            use_additional_convs: true,
            bias: true,
        }
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
        module.forward(
            get_tensor(input_reader, "c", device),
            None, // g
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
