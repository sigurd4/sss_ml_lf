use std::ops::AddAssign;

use num::Float;
use super::*;

#[derive(Clone, Copy, Debug)]
pub struct OneZeroLoss;

impl<F, const Y: usize> LossFunction<F, Y, Y, Y> for OneZeroLoss
where
    F: Float + AddAssign + Default
{
    fn lf_loss(&self, y_true: [F; Y], y_est: [F; Y]) -> [F; Y]
    {
        y_true.comap(y_est, |y_true, y_est| f!((y_est != y_true) as u8))
    }
    fn lf_loss_grad(&self, _y_true: [F; Y], _y_est: [F; Y]) -> [[F; Y]; Y]
    {
        [[F::zero(); Y]; Y]
    }
}

#[cfg(test)]
mod test
{
    use crate::tests as t;
    use super::OneZeroLoss as LF;

    #[test]
    fn test()
    {
        t::test(LF, [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 4.0, 5.0, 4.0]);
        t::graph_2d(LF, [0.3, -0.1], -1.0..1.0);
    }
}