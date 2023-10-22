use std::ops::AddAssign;

use num::Float;
use super::*;

#[derive(Clone, Copy, Debug)]
pub struct TwoParameterLoss;

impl<F, const N: usize> LossFunction<F, {2*N}, N, N> for TwoParameterLoss
where
    F: Float + AddAssign + Default,
    [(); 0 - 2*N % 2]:,
    [(); (2*N) / 2]:,
    [(); (2*N) / 2 - N]:,
    [(); N - (2*N) / 2]:,
    [(); N - N]:
{
    fn lf_loss(&self, y_true: [F; 2*N], y_est: [F; N]) -> [F; N]
    {
        y_true.array_chunks_exact()
            .reformulate_length()
            .comap(y_est, |[y_low, y_high], y_est| ((y_est - y_low).abs() + (y_est - y_high).abs() - (y_low - y_high))*f!(0.5))
    }
    fn lf_loss_grad(&self, y_true: [F; 2*N], y_est: [F; N]) -> [[F; N]; N]
    {
        y_true.array_chunks_exact()
            .reformulate_length()
            .comap(y_est, |[y_low, y_high], y_est| ((y_est - y_low).signum() + (y_est - y_high).signum())*f!(0.5))
            .diagonal()
    }
}

#[cfg(test)]
mod test
{
    use array_math::ArrayNdOps;

    use crate::tests as t;
    use super::TwoParameterLoss as LF;

    #[test]
    fn test()
    {
        t::test(LF,
            [[0.9, 1.0], [1.8, 2.0], [1.0, 3.0], [3.5, 4.0], [-1.0, 5.0]].flatten_nd_array(),
            [1.0, 2.0, 4.0, 5.0, 4.0]
        );
        t::graph_2d(LF, [[0.3, 0.4], [-0.1, 5.0]].flatten_nd_array(), -1.0..1.0);
    }
}