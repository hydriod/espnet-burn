mod dds;

use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn_store::PytorchStore;

pub use self::dds::{DilatedDepthSeparableConv, DilatedDepthSeparableConvConfig};

fn cumsum_4d<B: Backend>(x: Tensor<B, 4>, dim: usize) -> Tensor<B, 4> {
    let num_bins = x.dims()[dim];
    let mut slices = Vec::with_capacity(num_bins);
    let mut current = Tensor::zeros_like(&x.clone().slice_dim(dim, 0..1));
    for i in 0..num_bins {
        let slice = x.clone().slice_dim(dim, i..i + 1);
        current = current + slice;
        slices.push(current.clone());
    }
    Tensor::cat(slices, dim)
}

/// PyTorch `F.softplus(beta=1, threshold=20)` compatible softplus.
/// Burn's naive `log(1 + exp(beta*x)) / beta` overflows to inf for large x,
/// which then cascades into NaN in the spline inverse. Mirror PyTorch by
/// returning x directly where `beta*x > threshold`.
fn stable_softplus<B: Backend>(x: Tensor<B, 4>, beta: f64) -> Tensor<B, 4> {
    let bx = x.clone() * beta;
    let naive = (bx.clone().exp() + 1.0).log() / beta;
    naive.mask_where(bx.greater_equal_elem(20.0), x)
}

fn rational_quadratic_spline<B: Backend>(
    inputs: Tensor<B, 3>,
    unnormalized_widths: Tensor<B, 4>,
    unnormalized_heights: Tensor<B, 4>,
    unnormalized_derivatives: Tensor<B, 4>,
    inverse: bool,
    left: f64,
    right: f64,
    bottom: f64,
    top: f64,
    min_bin_width: f64,
    min_bin_height: f64,
    min_derivative: f64,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let num_bins = unnormalized_widths.dims()[3];
    let device = inputs.device();
    let [b_dim, c_dim, t_dim] = inputs.dims();

    let widths = softmax(unnormalized_widths, 3);
    let widths = widths * (1.0 - min_bin_width * num_bins as f64) + min_bin_width;
    let cumwidths = cumsum_4d(widths, 3);
    let pad_zero = Tensor::zeros([b_dim, c_dim, t_dim, 1], &device);
    let cumwidths = Tensor::cat(vec![pad_zero.clone(), cumwidths], 3);
    let cumwidths = cumwidths * (right - left) + left;

    let heights = burn::tensor::activation::softmax(unnormalized_heights, 3);
    let heights = heights * (1.0 - min_bin_height * num_bins as f64) + min_bin_height;
    let cumheights = cumsum_4d(heights, 3);
    let cumheights = Tensor::cat(vec![pad_zero, cumheights], 3);
    let cumheights = cumheights * (top - bottom) + bottom;

    let widths = cumwidths.clone().slice_dim(3, 1..num_bins + 1)
        - cumwidths.clone().slice_dim(3, 0..num_bins);
    let heights = cumheights.clone().slice_dim(3, 1..num_bins + 1)
        - cumheights.clone().slice_dim(3, 0..num_bins);

    let derivatives = stable_softplus(unnormalized_derivatives, 1.0) + min_derivative;

    let search_cum = if inverse { &cumheights } else { &cumwidths };
    let search_cum_eps = search_cum.clone();
    let num_cum_cols = search_cum_eps.dims()[3];
    let last_col = search_cum_eps
        .clone()
        .slice_dim(3, num_cum_cols - 1..num_cum_cols)
        + 1e-6;
    let search_cum_eps = Tensor::cat(
        vec![search_cum_eps.slice_dim(3, 0..num_cum_cols - 1), last_col],
        3,
    );

    let inputs_4d = inputs.clone().unsqueeze_dim(3);
    let bin_idx = inputs_4d
        .greater_equal(search_cum_eps)
        .int()
        .sum_dim(3)
        .sub_scalar(1)
        .clamp(0, (num_bins - 1) as i64);

    let input_cumwidths = cumwidths.gather(3, bin_idx.clone()).squeeze_dim(3);
    let input_bin_widths = widths.clone().gather(3, bin_idx.clone()).squeeze_dim(3);

    let input_cumheights = cumheights.gather(3, bin_idx.clone()).squeeze_dim(3);
    let delta = heights.clone() / widths;
    let input_delta = delta.gather(3, bin_idx.clone()).squeeze_dim(3);

    let input_derivatives = derivatives
        .clone()
        .gather(3, bin_idx.clone())
        .squeeze_dim(3);
    let input_derivatives_plus_one = derivatives
        .slice_dim(3, 1..num_bins + 1)
        .gather(3, bin_idx.clone())
        .squeeze_dim(3);

    let input_heights = heights.gather(3, bin_idx).squeeze_dim(3);

    if inverse {
        let diff = inputs - input_cumheights.clone();
        let term = input_derivatives.clone() + input_derivatives_plus_one.clone()
            - input_delta.clone() * 2.0;
        let a = diff.clone() * term.clone()
            + input_heights.clone() * (input_delta.clone() - input_derivatives.clone());
        let b = input_heights.clone() * input_derivatives.clone() - diff.clone() * term;
        let c = -input_delta.clone() * diff;

        let discriminant = b.clone().powf_scalar(2.0) - a.clone() * c.clone() * 4.0;
        let root = (c * 2.0) / (-b - discriminant.clamp_min(1e-12).sqrt());
        let outputs = root.clone() * input_bin_widths + input_cumwidths;

        let theta_one_minus_theta: Tensor<B, 3> = root.clone() * (1.0 - root.clone());
        let denominator = input_delta.clone()
            + (input_derivatives.clone() + input_derivatives_plus_one.clone()
                - input_delta.clone() * 2.0)
                * theta_one_minus_theta.clone();
        let one_minus_root: Tensor<B, 3> = 1.0 - root.clone();
        let derivative_numerator = input_delta.clone().powf_scalar(2.0)
            * (input_derivatives_plus_one * root.powf_scalar(2.0)
                + input_delta * 2.0 * theta_one_minus_theta
                + input_derivatives * one_minus_root.powf_scalar(2.0));
        let logabsdet = -(derivative_numerator.log() - denominator.log() * 2.0);

        (outputs, logabsdet)
    } else {
        let theta = (inputs - input_cumwidths) / input_bin_widths;
        let one_minus_theta: Tensor<B, 3> = 1.0 - theta.clone();
        let theta_one_minus_theta = theta.clone() * one_minus_theta.clone();

        let numerator = input_heights
            * (input_delta.clone() * theta.clone().powf_scalar(2.0)
                + input_derivatives.clone() * theta_one_minus_theta.clone());
        let denominator = input_delta.clone()
            + ((input_derivatives.clone() + input_derivatives_plus_one.clone()
                - input_delta.clone() * 2.0)
                * theta_one_minus_theta.clone());
        let outputs = input_cumheights + numerator / denominator.clone();

        let derivative_numerator = input_delta.clone().powf_scalar(2.0)
            * (input_derivatives_plus_one * theta.powf_scalar(2.0)
                + input_delta * 2.0 * theta_one_minus_theta
                + input_derivatives * one_minus_theta.powf_scalar(2.0));
        let logabsdet = derivative_numerator.log() - denominator.log() * 2.0;

        (outputs, logabsdet)
    }
}

fn piecewise_rational_quadratic_transform<B: Backend>(
    inputs: Tensor<B, 3>,
    unnormalized_widths: Tensor<B, 4>,
    unnormalized_heights: Tensor<B, 4>,
    unnormalized_derivatives: Tensor<B, 4>,
    inverse: bool,
    tail_bound: f64,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let min_bin_width = 1e-3;
    let min_bin_height = 1e-3;
    let min_derivative = 1e-3;

    let constant = (f64::exp(1.0 - min_derivative) - 1.0).ln();
    let [b, c, t, _] = unnormalized_derivatives.dims();
    let device = inputs.device();

    let pad_const = Tensor::full([b, c, t, 1], constant, &device);
    let unnormalized_derivatives = Tensor::cat(
        vec![pad_const.clone(), unnormalized_derivatives, pad_const],
        3,
    );

    let inside_mask = inputs
        .clone()
        .greater_equal_elem(-tail_bound)
        .bool_and(inputs.clone().lower_equal_elem(tail_bound));

    let inputs_clamped = inputs.clone().clamp(-tail_bound, tail_bound);

    let (outputs_spline, logabsdet_spline) = rational_quadratic_spline(
        inputs_clamped,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse,
        -tail_bound,
        tail_bound,
        -tail_bound,
        tail_bound,
        min_bin_width,
        min_bin_height,
        min_derivative,
    );

    let outputs = inputs
        .clone()
        .mask_where(inside_mask.clone(), outputs_spline);
    let logabsdet = Tensor::zeros_like(&inputs).mask_where(inside_mask, logabsdet_spline);

    (outputs, logabsdet)
}

/// Convolutional flow module.
#[derive(Module, Debug)]
pub struct ConvFlow<B: Backend> {
    half_channels: usize,
    hidden_channels: usize,
    bins: usize,
    tail_bound: f64,
    input_conv: Conv1d<B>,
    dds_conv: DilatedDepthSeparableConv<B>,
    proj: Conv1d<B>,
}

impl<B: Backend> ConvFlow<B> {
    pub fn new(
        in_channels: usize,
        hidden_channels: usize,
        kernel_size: usize,
        layers: usize,
        bins: Option<usize>,
        tail_bound: Option<f64>,
        device: &B::Device,
    ) -> Self {
        let bins = bins.unwrap_or(10);
        let tail_bound = tail_bound.unwrap_or(5.0);

        let half_channels = in_channels / 2;
        let input_conv = Conv1dConfig::new(half_channels, hidden_channels, 1).init(device);
        let dds_conv =
            DilatedDepthSeparableConvConfig::new(hidden_channels, kernel_size, layers).init(device);
        let out_channels = half_channels * (bins * 3 - 1);
        let proj = Conv1dConfig::new(hidden_channels, out_channels, 1).init(device);

        Self {
            half_channels,
            hidden_channels,
            bins,
            tail_bound,
            input_conv,
            dds_conv,
            proj,
        }
    }

    /// # Args:
    /// - `x`: Input tensor (B, channels, T).
    /// - `x_mask`: Mask tensor (B, 1, T).
    /// - `g`: Global conditioning tensor (B, channels, 1).
    /// - `inverse`: Whether to inverse the flow.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
        inverse: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let x = x.chunk(2, 1);
        let (xa, xb) = (x.get(0).unwrap().clone(), x.get(1).unwrap().clone());
        let h = self.input_conv.forward(xa.clone());
        let h = self.dds_conv.forward(h, x_mask.clone(), g);
        let h = self.proj.forward(h) * x_mask.clone();

        let [b, c, t] = xa.dims().map(|d| d as isize);

        // (B, half_channels, bins * 3 -1, T) -> (B, half_channels, T, bins * 3 - 1)
        let h = h.reshape([b, c, -1, t]).swap_dims(2, 3);

        let denom = (self.hidden_channels as f64).sqrt();
        let unnorm_widths = h.clone().slice_dim(3, 0..self.bins) / denom;
        let unnorm_heights = h.clone().slice_dim(3, self.bins..2 * self.bins) / denom;
        let unnorm_derivatives = h.slice_dim(3, 2 * self.bins..self.bins * 3 - 1);

        let (xb_out, logdet_abs) = piecewise_rational_quadratic_transform(
            xb,
            unnorm_widths,
            unnorm_heights,
            unnorm_derivatives,
            inverse,
            self.tail_bound,
        );

        let x_out = Tensor::cat(vec![xa, xb_out], 1) * x_mask.clone();
        let logdet = (logdet_abs * x_mask).sum_dims_squeeze(&[1, 2]);

        if !inverse {
            (x_out, logdet)
        } else {
            let device = x_out.device();
            (x_out, Tensor::zeros([b], &device))
        }
    }

    pub fn key_remap(&self, store: PytorchStore, parents: &Vec<&str>) -> PytorchStore {
        let mut parents = parents.clone();
        parents.push("dds_conv");
        self.dds_conv.key_remap(store, &parents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::Flex;
    use burn::backend::wgpu::Wgpu;
    use burn_store::{pytorch::PytorchReader, ModuleSnapshot, PytorchStore};

    const REL_TOL: f32 = 0.05;
    const FIXTURE_DIR: &str = "tests/fixtures/vits/generator/duration_predictor/flow_step/conv/mod";
    type ForwardOutput<B> = (Tensor<B, 3>, Tensor<B, 1>);
    type TestModule<B> = ConvFlow<B>;

    fn init_module<B: Backend>(device: &B::Device) -> TestModule<B> {
        ConvFlow::new(2, 192, 3, 3, None, None, device)
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
            true,
        )
    }

    /// 全要素が相対誤差 `tol` 以内かをチェックする。
    /// 期待値が0に近い要素は絶対誤差(ABS_EPS)を分母の下限にして判定する。
    fn assert_allclose<B: Backend>(actual: ForwardOutput<B>, expected: &PytorchReader, tol: f32) {
        // ↓ 出力の次元数(D)もPython側 run_forward() の出力shapeに合わせて置き換える
        let actual_tensor = actual.0;
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
