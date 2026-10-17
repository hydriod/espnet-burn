use burn::module::Param;
use burn::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

pub struct RelPositionMultiHeadedAttentionConfig {
    n_head: usize,
    n_feats: usize,
    dropout: f64,
}
impl RelPositionMultiHeadedAttentionConfig {
    pub fn new(n_head: usize, n_feats: usize, dropout: f64) -> Self {
        Self {
            n_head,
            n_feats,
            dropout,
        }
    }
    pub fn init<B: Backend>(&self, device: &B::Device) -> RelPositionMultiHeadedAttention<B> {
        let h = self.n_head as isize;
        let d_k = (self.n_feats / self.n_head) as isize;
        let linear_q = LinearConfig::new(self.n_feats, self.n_feats).init(device);
        let linear_k = LinearConfig::new(self.n_feats, self.n_feats).init(device);
        let linear_v = LinearConfig::new(self.n_feats, self.n_feats).init(device);
        let linear_out = LinearConfig::new(self.n_feats, self.n_feats).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();
        // linear transformation for positional encoding
        let linear_pos = LinearConfig::new(self.n_feats, self.n_feats)
            .with_bias(false)
            .init(device);
        // these two learnable bias are used in matrix c and matrix d
        // as described in https://arxiv.org/abs/1901.02860 Section 3.3
        let pos_bias_u = Initializer::XavierUniform { gain: 1.0 }.init([h, d_k], device);
        let pos_bias_v = Initializer::XavierUniform { gain: 1.0 }.init([h, d_k], device);

        RelPositionMultiHeadedAttention {
            linear_q,
            linear_k,
            linear_v,
            dropout,
            linear_out,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
            h,
            d_k,
        }
    }
}

#[derive(Module, Debug)]
pub struct RelPositionMultiHeadedAttention<B: Backend> {
    linear_q: Linear<B>,
    linear_k: Linear<B>,
    linear_v: Linear<B>,
    dropout: Dropout,
    linear_out: Linear<B>,
    linear_pos: Linear<B>,
    pos_bias_u: Param<Tensor<B, 2>>,
    pos_bias_v: Param<Tensor<B, 2>>,
    h: isize,
    d_k: isize,
}
impl<B: Backend> RelPositionMultiHeadedAttention<B> {
    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        pos_emb: Tensor<B, 3>,
        mask: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let (q, k, v) = self.forward_qkv(query, key, value);
        let q = q.swap_dims(1, 2); // (batch, time1, head, d_k)

        let [batch, _, head, d_k] = q.dims().map(|d| d as isize);
        let p = self
            .linear_pos
            .forward(pos_emb)
            .reshape([batch, -1, head, d_k])
            .swap_dims(1, 2); // (batch, head, 2*time1-1, d_k)

        // (batch, head, time1, d_k)
        let q_with_bias_u =
            (q.clone() + self.pos_bias_u.val().unsqueeze_dims(&[0, 0])).swap_dims(1, 2);
        // (batch, head, time1, d_k)
        let q_with_bias_v = (q + self.pos_bias_v.val().unsqueeze_dims(&[0, 0])).swap_dims(1, 2);

        // compute attention score
        // first compute matrix a and matrix c
        // as described in https://arxiv.org/abs/1901.02860 Section 3.3
        // (batch, head, time1, time2)
        let matrix_ac = Tensor::matmul(q_with_bias_u, k.swap_dims(-2, -1));

        // compute matrix b and matrix d
        // (batch, head, time1, 2*time1-1)
        let matrix_bd = Tensor::matmul(q_with_bias_v, p.swap_dims(-2, -1));
        let matrix_bd = self.rel_shift(matrix_bd);

        let scores = (matrix_ac + matrix_bd) / (d_k as f64).sqrt(); // (batch, head, time1, time2)

        self.forward_attention(v, scores, mask)
    }

    fn rel_shift(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n_batch, n_head, n_time, n_pos_emb] = x.dims().map(|d| d as isize);
        let zero_pad = Tensor::<B, 4>::zeros([n_batch, n_head, n_time, 1], &x.device());
        let x_padded = Tensor::cat(vec![zero_pad, x], 3);

        let x = x_padded
            .reshape([n_batch, n_head, -1, n_time])
            .slice_dim(2, 1..)
            .reshape([n_batch, n_head, n_time, n_pos_emb])
            .slice_dim(3, ..((n_pos_emb / 2) + 1)); // only keep the positions from 0 to time2

        x
    }

    fn forward_qkv(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let n_batch = query.dims()[0] as isize;
        let q = self
            .linear_q
            .forward(query)
            .reshape([n_batch, -1, self.h, self.d_k]);
        let k = self
            .linear_k
            .forward(key)
            .reshape([n_batch, -1, self.h, self.d_k]);
        let v = self
            .linear_v
            .forward(value)
            .reshape([n_batch, -1, self.h, self.d_k]);

        let q = q.swap_dims(1, 2); // (batch, head, time1, d_k)
        let k = k.swap_dims(1, 2); // (batch, head, time2, d_k)
        let v = v.swap_dims(1, 2); // (batch, head, time2, d_k)

        (q, k, v)
    }

    fn forward_attention(
        &self,
        value: Tensor<B, 4>,
        scores: Tensor<B, 4>,
        mask: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let min_value = f32::MIN;
        let mask = mask.unsqueeze_dim(1).equal_elem(0); // (batch, 1, *, time2)
        let scores = scores.mask_fill(mask.clone(), min_value);

        let attn = softmax(scores, 3).mask_fill(mask, 0.0); // (batch, head, time1, time2)

        let p_attn = self.dropout.forward(attn);
        let x = Tensor::matmul(p_attn, value); // (batch, head, time1, d_k)
        let [n_batch, n_head, _, d_k] = x.dims();
        let x = x
            .swap_dims(1, 2)
            .reshape([n_batch as isize, -1, (n_head * d_k) as isize]); // (batch, time1, d_model)

        self.linear_out.forward(x) // (batch, time1, d_model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::Flex;
    use burn::backend::wgpu::Wgpu;
    use burn_store::{pytorch::PytorchReader, ModuleSnapshot, PytorchStore};

    const REL_TOL: f32 = 0.05;
    const FIXTURE_DIR: &str = "tests/fixtures/vits/generator/text_encoder/encoder/layer/attention";
    type ForwardOutput<B> = Tensor<B, 3>;
    type TestModule<B> = RelPositionMultiHeadedAttention<B>;

    fn init_module<B: Backend>(device: &B::Device) -> TestModule<B> {
        RelPositionMultiHeadedAttentionConfig::new(2, 192, 0.1).init(device)
    }

    fn key_remap<B: Backend>(store: PytorchStore, _: &TestModule<B>) -> PytorchStore {
        // let store = module.key_remap(store, &vec![]);
        store
    }
    fn forward<B: Backend>(
        module: &TestModule<B>,
        input_reader: &PytorchReader,
        device: &B::Device,
    ) -> ForwardOutput<B> {
        module.forward(
            get_tensor(input_reader, "query", device),
            get_tensor(input_reader, "key", device),
            get_tensor(input_reader, "value", device),
            get_tensor(input_reader, "pos_emb", device),
            get_tensor(input_reader, "mask", device),
        )
    }

    /// 全要素が相対誤差 `tol` 以内かをチェックする。
    /// 期待値が0に近い要素は絶対誤差(ABS_EPS)を分母の下限にして判定する。
    fn assert_allclose<B: Backend>(actual: ForwardOutput<B>, expected: &PytorchReader, tol: f32) {
        // ↓ 出力の次元数(D)もPython側 run_forward() の出力shapeに合わせて置き換える
        let actual_output_0 = actual;
        let expected_output_0 = get_tensor(expected, "output_0", &Default::default());

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

    #[test]
    #[ignore = "デバッグ用"]
    fn test_inspect_forward() {
        type B = Wgpu;
        let device = Default::default();
        let mut module = init_module(&device);

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

        let query = get_tensor(&input_reader, "query", &device);
        let key = get_tensor(&input_reader, "key", &device);
        let value = get_tensor(&input_reader, "value", &device);
        let pos_emb = get_tensor::<B, 3>(&input_reader, "pos_emb", &device);
        let mask = get_tensor(&input_reader, "mask", &device);

        // --- 4. forward
        let (q, k, v) = module.forward_qkv(query, key, value);
        trace_tensor("q_after_forward_qkv", q.clone());
        trace_tensor("k_after_forward_qkv", k.clone());
        trace_tensor("v_after_forward_qkv", v.clone());
        let q = q.swap_dims(1, 2); // (batch, time1, head, d_k)
        trace_tensor("q_after_swap", q.clone());

        let [batch, _, head, d_k] = q.dims().map(|d| d as isize);
        let p = module
            .linear_pos
            .forward(pos_emb)
            .reshape([batch, -1, head, d_k])
            .swap_dims(1, 2); // (batch, head, 2*time1-1, d_k)
        trace_tensor("p_after_linear_pos", p.clone());

        // (batch, head, time1, d_k)
        let q_with_bias_u =
            (q.clone() + module.pos_bias_u.val().unsqueeze_dims(&[0, 0])).swap_dims(1, 2);
        trace_tensor("q_with_bias_u", q_with_bias_u.clone());
        // (batch, head, time1, d_k)
        let q_with_bias_v = (q + module.pos_bias_v.val().unsqueeze_dims(&[0, 0])).swap_dims(1, 2);
        trace_tensor("q_with_bias_v", q_with_bias_v.clone());

        // compute attention score
        // first compute matrix a and matrix c
        // as described in https://arxiv.org/abs/1901.02860 Section 3.3
        // (batch, head, time1, time2)
        let matrix_ac = Tensor::matmul(q_with_bias_u, k.swap_dims(-2, -1));
        trace_tensor("matrix_ac", matrix_ac.clone());

        // compute matrix b and matrix d
        // (batch, head, time1, 2*time1-1)
        let matrix_bd = Tensor::matmul(q_with_bias_v, p.swap_dims(-2, -1));
        trace_tensor("matrix_bd_before_rel_shift", matrix_bd.clone());
        let matrix_bd = module.rel_shift(matrix_bd);
        trace_tensor("matrix_bd_after_rel_shift", matrix_bd.clone());

        let scores = (matrix_ac + matrix_bd) / (d_k as f64).sqrt(); // (batch, head, time1, time2)
        trace_tensor("scores", scores.clone());

        let result = module.forward_attention(v, scores, mask);
        trace_tensor("result", result);
    }

    fn trace_tensor<B: Backend, const D: usize>(name: &str, tensor: Tensor<B, D>) {
        let values = to_vec(tensor.clone());
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let preview = values.iter().take(8).copied().collect::<Vec<_>>();

        println!(
            "{name}: shape={:?}, min={min:.7}, max={max:.7}, mean={mean:.7}\n\tfirst8={preview:?}",
            tensor.dims()
        );
    }

    #[ignore = "Wgpuのmatmul挙動確認用"]
    #[test]
    fn test_slice_contigous() {
        type B = Flex;
        let device = Default::default();
        let mut module = init_module::<B>(&device);

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

        let key = get_tensor::<B, 3>(&input_reader, "key", &device);

        let key_first_two_rows = key.slice_dim(1, 0..2);
        let weight_first_two_columns = module.linear_k.weight.val().detach().slice_dim(1, 0..2);
        let weight_first_two_columns = weight_first_two_columns.to_data();
        let weight_first_two_columns = Tensor::<B, 2>::from_data(weight_first_two_columns, &device);
        let key_first_two_rows = key_first_two_rows.to_data();
        let key_first_two_rows = Tensor::<B, 3>::from_data(key_first_two_rows, &device);
        println!("key_first_two_rows: {key_first_two_rows}");
        println!("weight_first_two_columns: {weight_first_two_columns}");
        println!(
            "{}",
            key_first_two_rows.matmul(weight_first_two_columns.unsqueeze())
        );
    }
}
