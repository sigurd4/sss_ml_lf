use std::ops::AddAssign;

use num::Float;

use super::*;

#[derive(Clone, Copy, Debug)]
pub struct ExponentialLoss;

impl<F, const Y: usize> LossFunction<F, Y, Y, Y> for ExponentialLoss
where
    F: Float + AddAssign + Default,
    [(); Y - Y]:
{
    fn lf_loss(&self, y_true: [F; Y], y_est: [F; Y]) -> [F; Y]
    {
        y_true.comap(y_est, |y_true, y_est| (-y_true*y_est).exp())
    }
    fn lf_loss_grad(&self, y_true: [F; Y], y_est: [F; Y]) -> [[F; Y]; Y]
    {
        y_true.comap(y_est, |y_true, y_est| -y_true*(-y_true*y_est).exp())
            .diagonal()
    }
}

#[cfg(test)]
mod test
{
    use crate::tests as t;
    use super::ExponentialLoss as LF;

    #[test]
    fn test()
    {
        t::test(LF, [1.0, -1.0, 1.0, 1.0, -1.0], [0.6, -1.2, 0.8, 1.5, -1.3]);
        t::graph_2d(LF, [0.8, 0.2], -1.0..1.0);
    }
}