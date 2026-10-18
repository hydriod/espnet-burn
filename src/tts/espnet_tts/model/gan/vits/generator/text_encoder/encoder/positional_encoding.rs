use burn::nn::{Dropout, DropoutConfig};
use burn::prelude::*;

pub struct RelPositionalEncodingConfig {
    d_model: usize,
    dropout_rate: f64,
    // max_len: usize,
}
impl RelPositionalEncodingConfig {
    pub fn new(d_model: usize, dropout_rate: f64) -> Self {
        Self {
            d_model,
            dropout_rate,
        }
    }

    pub fn init(&self) -> RelPositionalEncoding {
        RelPositionalEncoding {
            dropout: DropoutConfig::new(self.dropout_rate).init(),
            d_model: self.d_model as i64,
        }
    }
}

#[derive(Module, Debug, Clone)]
pub struct RelPositionalEncoding {
    dropout: Dropout,
    d_model: i64,
    // max_len: i64,
}
impl RelPositionalEncoding {
    fn extend_pe<B: Backend>(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, n_time, _] = x.dims().map(|d| d as i64);
        let position = Tensor::arange(0..n_time, &x.device())
            .float()
            .reshape([-1, 1]);
        let div_term = Tensor::arange_step(0..self.d_model, 2, &x.device())
            .float()
            .mul_scalar(-f64::ln(10000.0) / self.d_model as f64)
            .exp()
            .reshape([1, -1]);

        let pe_positive_sin = position.clone().matmul(div_term.clone()).sin();
        let pe_positive_cos = position.clone().matmul(div_term.clone()).cos();
        let pe_negative_sin = position.clone().matmul(-div_term.clone()).sin();
        let pe_negative_cos = position.clone().matmul(-div_term.clone()).cos();

        let pe_positive = Tensor::stack::<3>(vec![pe_positive_sin, pe_positive_cos], 2)
            .reshape([n_time, -1])
            .flip([0]);
        let pe_negative = Tensor::stack::<3>(vec![pe_negative_sin, pe_negative_cos], 2)
            .reshape([n_time, -1])
            .slice_dim(0, 1..);
        let pe = Tensor::cat(vec![pe_positive, pe_negative], 0).reshape([1, -1, self.d_model]);

        pe
    }

    pub fn forward<B: Backend>(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let x = x * f64::sqrt(self.d_model as f64);
        let pe = self.extend_pe(&x);
        (self.dropout.forward(x), self.dropout.forward(pe))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::Flex;
    use burn::backend::wgpu::Wgpu;
    use burn_store::pytorch::PytorchReader;

    const REL_TOL: f32 = 0.05;
    const FIXTURE_DIR: &str =
        "tests/fixtures/vits/generator/text_encoder/encoder/positional_encoding";
    type ForwardOutput<B> = (Tensor<B, 3>, Tensor<B, 3>);
    type TestModule = RelPositionalEncoding;

    fn init_module<B: Backend>(_device: &B::Device) -> TestModule {
        RelPositionalEncodingConfig::new(192, 0.0).init()
    }

    fn forward<B: Backend>(
        module: &TestModule,
        input_reader: &PytorchReader,
        device: &B::Device,
    ) -> ForwardOutput<B> {
        module.forward(get_tensor(input_reader, "x", device))
    }

    /// 全要素が相対誤差 `tol` 以内かをチェックする。
    /// 期待値が0に近い要素は絶対誤差(ABS_EPS)を分母の下限にして判定する。
    fn assert_allclose<B: Backend>(actual: ForwardOutput<B>, expected: &PytorchReader, tol: f32) {
        // ↓ 出力の次元数(D)もPython側 run_forward() の出力shapeに合わせて置き換える
        let actual_tensor = actual.0;
        let expected_tensor = get_tensor(expected, "output_0", &Default::default());
        assert_close(actual_tensor, expected_tensor, tol);

        let actual_tensor = actual.1;
        let expected_tensor = get_tensor(expected, "output_1", &Default::default());
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
        let module = init_module::<TestBackend>(&device);

        // --- 3. Python側で生成した入力をロードする ---
        let input_path = format!("{FIXTURE_DIR}/input.pth");
        let input_reader = PytorchReader::new(&input_path)
            .unwrap_or_else(|e| panic!("input.pth の読み込みに失敗しました:\n{e}"));

        // --- 4. Burn側 forward を実行する ---
        let actual_output = forward::<TestBackend>(&module, &input_reader, &device);

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
