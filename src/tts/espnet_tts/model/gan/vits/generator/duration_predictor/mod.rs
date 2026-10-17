mod flow_step;
mod log_flow;

use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::prelude::*;
use burn::tensor::activation::{log_sigmoid, sigmoid};
use burn::tensor::Distribution;
use burn_store::PytorchStore;

use crate::error::Error;

use self::flow_step::{DilatedDepthSeparableConv, DilatedDepthSeparableConvConfig, FlowStep};
use self::log_flow::LogFlow;

pub struct StochasticDurationPredictorConfig {
    pub channels: usize,
    pub kernel_size: usize,
    pub dropout_rate: f64,
    pub flows: usize,
    pub dds_conv_layers: usize,
    pub global_channels: usize,
}

impl StochasticDurationPredictorConfig {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        dropout_rate: f64,
        flows: usize,
        dds_conv_layers: usize,
        global_channels: usize,
    ) -> Self {
        Self {
            channels,
            kernel_size,
            dropout_rate,
            flows,
            dds_conv_layers,
            global_channels,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> StochasticDurationPredictor<B> {
        let pre = Conv1dConfig::new(self.channels, self.channels, 1).init(device);
        let dds = DilatedDepthSeparableConvConfig::new(
            self.channels,
            self.kernel_size,
            self.dds_conv_layers,
        )
        .with_dropout_rate(self.dropout_rate)
        .init(device);
        let proj = Conv1dConfig::new(self.channels, self.channels, 1).init(device);

        let log_flow = LogFlow;

        let mut flows_list = Vec::new();
        flows_list.push(FlowStep::elementwise_affine_flow(2, device));
        for _ in 0..self.flows {
            flows_list.push(FlowStep::conv_flow(
                2,
                self.channels,
                self.kernel_size,
                self.dds_conv_layers,
                Some(10),
                Some(5.0),
                device,
            ));
            flows_list.push(FlowStep::flip_flow());
        }

        let post_pre = Conv1dConfig::new(1, self.channels, 1).init(device);
        let post_dds = DilatedDepthSeparableConvConfig::new(
            self.channels,
            self.kernel_size,
            self.dds_conv_layers,
        )
        .with_dropout_rate(self.dropout_rate)
        .init(device);
        let post_proj = Conv1dConfig::new(self.channels, self.channels, 1).init(device);

        let mut post_flows_list = Vec::new();
        post_flows_list.push(FlowStep::elementwise_affine_flow(2, device));
        for _ in 0..self.flows {
            post_flows_list.push(FlowStep::conv_flow(
                2,
                self.channels,
                self.kernel_size,
                self.dds_conv_layers,
                Some(10),
                Some(5.0),
                device,
            ));
            post_flows_list.push(FlowStep::flip_flow());
        }

        let global_conv = if self.global_channels > 0 {
            Some(Conv1dConfig::new(self.global_channels, self.channels, 1).init(device))
        } else {
            None
        };

        StochasticDurationPredictor {
            pre,
            dds,
            proj,
            log_flow,
            flows: flows_list,
            post_pre,
            post_dds,
            post_proj,
            post_flows: post_flows_list,
            global_conv,
        }
    }
}

#[derive(Module, Debug)]
pub struct StochasticDurationPredictor<B: Backend> {
    pre: Conv1d<B>,
    dds: DilatedDepthSeparableConv<B>,
    proj: Conv1d<B>,
    log_flow: LogFlow,
    flows: Vec<FlowStep<B>>,
    post_pre: Conv1d<B>,
    post_dds: DilatedDepthSeparableConv<B>,
    post_proj: Conv1d<B>,
    post_flows: Vec<FlowStep<B>>,
    global_conv: Option<Conv1d<B>>,
}

impl<B: Backend> StochasticDurationPredictor<B> {
    /// # Args
    /// - `x`: Input tensor (B, channels, T_text).
    /// - `x_mask`: Mask tensor (B, 1, T_text).
    /// - `w`: Duration tensor (B, 1, T_text).
    /// - `g`: Global conditioning tensor (B, global_channels, 1).
    /// - `inverse`: Whether to inverse the flow.
    /// - `noise_scale`: Noise scale value.
    /// # Returns
    /// If `not inverse`, negative log-likelihood (NLL) tensor (B, 1, 1).
    /// If `inverse`, log-duration tensor (B, 1, T_text).
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        w: Option<Tensor<B, 3>>,
        g: Option<Tensor<B, 3>>,
        inverse: bool,
        noise_scale: f64,
    ) -> Result<Tensor<B, 3>, Error> {
        let x = self.pre.forward(x);
        let x = match (&self.global_conv, g) {
            (Some(conv), Some(g_tensor)) => x + conv.forward(g_tensor),
            _ => x,
        };
        let x = self.dds.forward(x, x_mask.clone(), None);
        let x = self.proj.forward(x) * x_mask.clone();

        if !inverse {
            let w = w.ok_or(Error::Argument(
                "w must be provided when inverse is false.".to_string(),
            ))?;
            let [b, _, t] = w.dims();
            let h_w = self.post_pre.forward(w.clone());
            let h_w = self.post_dds.forward(h_w, x_mask.clone(), None);
            let h_w = self.post_proj.forward(h_w) * x_mask.clone();

            let e_q = Tensor::random([b, 2, t], Distribution::Normal(0.0, 1.0), &x.device())
                * x_mask.clone();
            // initialize
            let mut z_q = e_q.clone();
            let mut logdet_tot_q = Tensor::zeros([b], &x.device());
            let g_cond = x.clone() + h_w;

            for flow in &self.post_flows {
                let (z_q_next, logdet_q) =
                    flow.forward(z_q, x_mask.clone(), Some(g_cond.clone()), false);
                z_q = z_q_next;
                logdet_tot_q = logdet_tot_q + logdet_q;
            }

            let z_q = z_q.chunk(2, 1);
            let (z_u, z1) = (z_q.get(0).unwrap().clone(), z_q.get(1).unwrap().clone());

            let u = sigmoid(z_u.clone()) * x_mask.clone();
            let z0 = (w - u) * x_mask.clone();

            let log_sig_pos = log_sigmoid(z_u.clone());
            let log_sig_neg = log_sigmoid(-z_u);
            logdet_tot_q = logdet_tot_q
                + ((log_sig_pos + log_sig_neg) * x_mask.clone()).sum_dims_squeeze(&[1, 2]);

            let const_log2pi = (2.0 * std::f64::consts::PI).ln();
            let logq = ((const_log2pi + e_q.powf_scalar(2.0)) * -0.5 * x_mask.clone())
                .sum_dims_squeeze(&[1, 2])
                - logdet_tot_q;

            let (z0, logdet) = self.log_flow.forward(z0, x_mask.clone(), false);
            let mut logdet_tot = logdet;
            let mut z = Tensor::cat(vec![z0, z1], 1);

            for flow in &self.flows {
                let (z_next, logdet) = flow.forward(z, x_mask.clone(), Some(x.clone()), false);
                z = z_next;
                logdet_tot = logdet_tot + logdet;
            }

            let nll = ((const_log2pi + z.powf_scalar(2.0)) * 0.5 * x_mask)
                .sum_dims_squeeze(&[1, 2])
                - logdet_tot;

            let res = nll + logq;
            Ok(res.unsqueeze_dims(&[1, 2]))
        } else {
            let mut rev_flows: Vec<&FlowStep<B>> = self.flows.iter().rev().collect();
            if rev_flows.len() >= 2 {
                let last = rev_flows.pop().unwrap();
                rev_flows.pop();
                rev_flows.push(last);
            }

            let [b, _, t] = x.dims();
            let mut z = Tensor::random([b, 2, t], Distribution::Normal(0.0, 1.0), &x.device())
                * noise_scale;

            for flow in rev_flows {
                let (z_next, _) = flow.forward(z, x_mask.clone(), Some(x.clone()), true);
                z = z_next;
            }

            let z0 = z.slice_dim(1, ..1);
            Ok(z0)
        }
    }

    pub fn key_remap(&self, store: PytorchStore, parents: &Vec<&str>) -> PytorchStore {
        let mut parents = parents.clone();
        parents.push("dds");
        self.dds.key_remap(store, &parents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::Flex;
    use burn::backend::wgpu::Wgpu;
    use burn_store::{pytorch::PytorchReader, ModuleSnapshot, PytorchStore};

    const REL_TOL: f32 = 0.05;
    const FIXTURE_DIR: &str = "tests/fixtures/vits/generator/duration_predictor/mod";
    type ForwardOutput<B> = Tensor<B, 3>;
    type TestModule<B> = StochasticDurationPredictor<B>;

    fn init_module<B: Backend>(device: &B::Device) -> TestModule<B> {
        StochasticDurationPredictorConfig::new(192, 3, 0.5, 4, 3, 0).init(device)
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
        // forward内で生成される乱数の影響を消すためnoise_scaleを0.0に設定
        module
            .forward(
                get_tensor(input_reader, "x", device),
                get_tensor(input_reader, "x_mask", device),
                None,
                None,
                true,
                0.0,
            )
            .unwrap_or_else(|e| panic!("forwardに失敗。\n{e}"))
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
