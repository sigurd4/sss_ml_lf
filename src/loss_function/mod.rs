use super::*;

moddef::moddef!(
    flat(pub) mod {
        square_error_loss,
        one_zero_loss,
        two_parameter_loss,

        mean_square_error,
        mean_absolute_error,
        mean_bias_error,

        binary_cross_entropy_loss,
        multi_class_svm_loss
    }
);

pub trait LossFunction<F, const Y_TRUE: usize, const Y_EST: usize, const L: usize = 1>
{
    fn lf_loss(&self, y_true: [F; Y_TRUE], y_est: [F; Y_EST]) -> [F; L];
    fn lf_loss_grad(&self, y_true: [F; Y_TRUE], y_est: [F; Y_EST]) -> [[F; Y_EST]; L];
}