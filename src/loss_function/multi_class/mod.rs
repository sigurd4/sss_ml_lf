moddef::moddef!(
    flat(pub) mod {
        exponential_loss,
        logistic_loss,
        square_loss,
        savage_loss,
        tangent_loss,
        hinge_loss,

        generalized_smooth_hinge_loss
    }
);

use super::*;