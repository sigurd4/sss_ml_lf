use std::ops::AddAssign;

use num::Float;
use num_identities_const::ZeroConst;

use super::*;

#[derive(Clone, Copy, Debug)]
pub struct BinaryCrossEntropyLoss;

impl<F, const Y: usize> LossFunction<F, Y, Y> for BinaryCrossEntropyLoss
where
    F: Float + AddAssign + ZeroConst
{
    fn lf_loss(&self, y_true: [F; Y], y_est: [F; Y]) -> [F; 1]
    {
        let len_inv = f!(Y; F).recip();
        
        [
            -len_inv*y_true.comap(y_est, |y_true, y_est| match (y_true.is_zero(), y_true.is_one())
                {
                    (false, false) => y_true*y_est.ln() + (F::one() - y_true)*(-y_est).ln_1p(),
                    (false, true) => y_true*y_est.ln(),
                    (true, false) => (F::one() - y_true)*(-y_est).ln_1p(),
                    (true, true) => F::zero()
                }).sum()
        ]
    }
    fn lf_loss_grad(&self, y_true: [F; Y], y_est: [F; Y]) -> [[F; Y]; 1]
    {
        let len_inv = f!(Y; F).recip();

        [
            y_true.comap(y_est, |y_true, y_est| -len_inv*match (y_true.is_zero(), y_true.is_one())
                {
                    (false, false) => y_true/y_est - (F::one() - y_true)/(F::one() - y_est),
                    (false, true) => y_true/y_est,
                    (true, false) => -(F::one() - y_true)/(F::one() - y_est),
                    (true, true) => F::zero()
                })
        ]
    }
}

#[cfg(test)]
mod test
{
    use crate::tests as t;
    use super::BinaryCrossEntropyLoss as LF;

    #[test]
    fn test()
    {
        t::test(LF, [1.0, 0.0, 1.0, 0.0, 1.0], [0.9, 0.1, 0.9, 0.1, 0.9]);
        t::graph_2d(LF, [0.2, 0.0], 0.001..1.0);
    }
}