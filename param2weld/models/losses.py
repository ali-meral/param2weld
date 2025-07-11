import tensorflow as tf


def mae_loss(y_true, y_pred):
    """Mean Absolute Error loss."""
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def ssim_loss_term(y_true, y_pred):
    """
    Structural Similarity Index Metric (SSIM).
    Used as a metric for logging.
    """
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def ssim_loss(y_true, y_pred):
    """
    SSIM loss used in training — 1 - SSIM, lower is better.
    """
    return 1.0 - ssim_loss_term(y_true, y_pred)


def hybrid_loss(w_mae=0.8, w_ssim=0.2):
    """
    Return a custom hybrid loss function combining MAE and SSIM.

    Parameters
    ----------
    w_mae : float
        Weight for the MAE component.
    w_ssim : float
        Weight for the SSIM component.

    Returns
    -------
    function
        A loss function that computes weighted MAE + SSIM loss.
    """

    def loss_fn(y_true, y_pred):
        return w_mae * mae_loss(y_true, y_pred) + w_ssim * ssim_loss(y_true, y_pred)

    return loss_fn
