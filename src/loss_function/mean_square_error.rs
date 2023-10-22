use std::ops::AddAssign;

use num::Float;
use num_identities_const::ZeroConst;
use super::*;

#[derive(Clone, Copy, Debug)]
pub struct MeanSquareError;

impl<F, const Y: usize> LossFunction<F, Y, Y> for MeanSquareError
where
    F: Float + AddAssign + ZeroConst
{
    fn lf_loss(&self, y_true: [F; Y], y_est: [F; Y]) -> [F; 1]
    {
        let len_inv = f!(Y; F).recip();

        [
            len_inv*y_true.comap(y_est, |y_true, y_est| (y_est - y_true)*(y_est - y_true))
                .sum()
        ]
    }
    fn lf_loss_grad(&self, y_true: [F; Y], y_est: [F; Y]) -> [[F; Y]; 1]
    {
        let len_inv2 = f!(2.0; F)/f!(Y);

        [
            y_true.comap(y_est, |y_true, y_est| len_inv2*(y_est - y_true))
        ]
    }
}

#[cfg(test)]
mod test
{
    use crate::tests as t;
    use super::MeanSquareError as LF;

    #[test]
    fn test()
    {
        t::test(LF, [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 4.0, 5.0, 4.0]);
        t::graph_2d(LF, [0.0, 0.0], -1.0..1.0);
    }
}