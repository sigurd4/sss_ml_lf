use num::Float;
use super::*;

#[derive(Clone, Copy, Debug)]
pub struct SquareErrorLoss;

impl<F, const N: usize> LossFunction<F, N, N, N> for SquareErrorLoss
where
    F: Float + Default,
    [(); N - N]:
{
    fn lf_loss(&self, y_true: [F; N], y_est: [F; N]) -> [F; N]
    {
        y_true.comap(y_est, |y_true, y_est| {
            let e = y_est - y_true;
            e*e
        })
    }
    fn lf_loss_grad(&self, y_true: [F; N], y_est: [F; N]) -> [[F; N]; N]
    {
        y_true.comap(y_est, |y_true, y_est| (y_est - y_true)*f!(2.0))
            .diagonal()
    }
}

#[cfg(test)]
mod test
{
    use crate::tests as t;
    use super::SquareErrorLoss as LF;

    #[test]
    fn test()
    {
        t::test(LF, [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 4.0, 5.0, 4.0]);
        t::graph_2d(LF, [0.3, -0.1], -1.0..1.0);
    }
}