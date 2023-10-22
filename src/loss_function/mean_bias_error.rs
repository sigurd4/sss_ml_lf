use std::ops::AddAssign;

use num::Float;
use num_identities_const::ZeroConst;

use super::*;

#[derive(Clone, Copy, Debug)]
pub struct MeanBiasError;

impl<F, const Y: usize> LossFunction<F, Y, Y> for MeanBiasError
where
    F: Float + AddAssign + ZeroConst
{
    fn lf_loss(&self, y_true: [F; Y], y_est: [F; Y]) -> [F; 1]
    {
        let len_inv = f!(Y; F).recip();
        
        [
            len_inv*y_true.comap(y_est, |y_true, y_est| y_est - y_true)
                .sum()
        ]
    }
    fn lf_loss_grad(&self, _y_true: [F; Y], _y_est: [F; Y]) -> [[F; Y]; 1]
    {
        let len_inv = f!(Y; F).recip();

        [[len_inv; Y]]
    }
}

#[cfg(test)]
mod test
{
    use crate::tests as t;
    use super::MeanBiasError as LF;

    #[test]
    fn test()
    {
        t::test(LF, [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 4.0, 5.0, 4.0]);
        t::graph_2d(LF, [0.3, -0.1], -1.0..1.0);
    }
}