import tensorflow as tf

from .instrument import BGM_IV
from ..networks import DemandImageCovariateDecoder, DemandImageEncoder

_SIGMA_TIME = 0.1


class BGM_IV_Image(BGM_IV):
    """BGM-IV with convolutional covariate networks for image covariates."""

    def __init__(
        self,
        params,
        timestamp=None,
        random_seed=None,
        auto_restore_checkpoint=True,
    ):
        params = dict(params)
        if int(params.get("v_dim", 0)) != 785:
            raise ValueError("`BGM_IV_Image` requires `v_dim == 785` (time + 28x28 image).")
        if int(params.get("w_dim", 0)) != 1:
            raise ValueError("`BGM_IV_Image` requires `w_dim == 1`.")

        super().__init__(
            params=params,
            timestamp=timestamp,
            random_seed=random_seed,
            auto_restore_checkpoint=False,
        )

        z_dim = sum(self.params["z_dims"])
        self.e_net = DemandImageEncoder(
            z_dim=z_dim,
            v_dim=self.params["v_dim"],
            name="e_net",
        )
        self.g_net = DemandImageCovariateDecoder(
            z_dim=z_dim,
            v_dim=self.params["v_dim"],
            name="g_net",
        )
        self.initialize_nets()
        # Non-fused BatchNormalization: the fused GPU gradient is not deterministic.
        self._apply_bn_determinism()

        self.ckpt = tf.train.Checkpoint(
            g_net=self.g_net,
            e_net=self.e_net,
            f_net=self.f_net,
            h_net=self.h_net,
            dz_net=self.dz_net,
            egm_sigma2_x_ema=self.egm_sigma2_x_ema,
            g_pre_optimizer=self.g_pre_optimizer,
            d_pre_optimizer=self.d_pre_optimizer,
            g_optimizer=self.g_optimizer,
            f_optimizer=self.f_optimizer,
            h_optimizer=self.h_optimizer,
            posterior_optimizer=self.posterior_optimizer,
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.checkpoint_path, max_to_keep=5
        )
        if auto_restore_checkpoint and self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!!")

    def initialize_nets(self):
        z_dim = sum(self.params["z_dims"])
        z0_dim = self.params["z_dims"][0]
        z1_dim = self.params["z_dims"][1]
        z2_dim = self.params["z_dims"][2]

        self.g_net(tf.zeros((1, z_dim), dtype=tf.float32))
        self.e_net(tf.zeros((1, self.params["v_dim"]), dtype=tf.float32))
        self.f_net(tf.zeros((1, z0_dim + z1_dim + 1), dtype=tf.float32))
        self.h_net(
            tf.zeros((1, z0_dim + z2_dim + self.params["w_dim"]), dtype=tf.float32)
        )

    @staticmethod
    def _split_public_covariates(data_v):
        data_v = tf.cast(data_v, tf.float32)
        time = data_v[:, :1]
        image_norm = data_v[:, 1:785] / 255.0
        return time, image_norm

    def _decode_covariates(self, data_z, training=True):
        decoded = dict(self.g_net(data_z, training=training))
        decoded["time_var"] = tf.ones_like(decoded["time_var"]) * tf.cast(
            _SIGMA_TIME ** 2, tf.float32
        )
        return decoded

    def _covariate_loss_terms(self, data_v, data_z, training=True):
        time_obs, image_obs = self._split_public_covariates(data_v)
        decoded = self._decode_covariates(data_z, training=training)

        time_mean = decoded["time_mean"]
        time_var = decoded["time_var"]
        image_logits = decoded["image_logits"]
        image_probs = decoded["image_probs"]

        time_nll = tf.squeeze(
            ((time_obs - time_mean) ** 2) / (2.0 * time_var)
            + 0.5 * tf.math.log(time_var),
            axis=1,
        )
        image_nll = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=image_obs,
                logits=image_logits,
            ),
            axis=1,
        )
        mse_time = tf.reduce_mean((time_obs - time_mean) ** 2)
        mse_image = tf.reduce_mean((image_obs - image_probs) ** 2)
        mse_v = tf.add_n([mse_time, mse_image]) / 2.0
        return tf.add_n([time_nll, image_nll]), mse_v, decoded

    def _covariate_cycle_mse(self, observed_v, reconstructed_v):
        observed_time, observed_image = self._split_public_covariates(observed_v)
        reconstructed_time, reconstructed_image = self._split_public_covariates(
            reconstructed_v
        )
        mse_time = tf.reduce_mean((observed_time - reconstructed_time) ** 2)
        mse_image = tf.reduce_mean((observed_image - reconstructed_image) ** 2)
        return tf.add_n([mse_time, mse_image]) / 2.0

    def _covariate_nll(self, data_v, data_z, training, eps=1e-6):
        del eps
        return self._covariate_loss_terms(data_v, data_z, training=training)[0]

    def _covariate_train_loss(self, data_z, data_v, eps=1e-6):
        del eps
        loss_terms, _, _ = self._covariate_loss_terms(data_v, data_z, training=True)
        return tf.reduce_mean(loss_terms)

    def _covariate_reconstruction(self, data_z):
        return self._decode_covariates(data_z, training=False)["public_v"]

    def _egm_covariate_block(self, data_z, data_v):
        decoded = self._decode_covariates(data_z, training=True)
        data_v_ = decoded["public_v"]

        data_z_ = self.e_net(data_v, training=True)
        data_z0, data_z1, data_z2 = self._split_z(data_z_)
        data_z__ = self.e_net(data_v_, training=True)
        data_v__ = self._decode_covariates(data_z_, training=True)["public_v"]
        sigma_square_loss = tf.reduce_mean(tf.square(decoded["time_var"]))
        return sigma_square_loss, data_z_, (data_z0, data_z1, data_z2), data_z__, data_v__
