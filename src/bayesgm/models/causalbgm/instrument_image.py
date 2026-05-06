import tensorflow as tf
import tensorflow_probability as tfp

from .instrument import CausalBGM_IV
from ..networks import DemandImageCovariateDecoder, DemandImageEncoder


class CausalBGM_IV_Image(CausalBGM_IV):
    """Image-aware IV model for the MNIST demand-design benchmark."""

    def __init__(self, params, timestamp=None, random_seed=None):
        params = dict(params)
        if int(params.get("v_dim", 0)) < 785:
            raise ValueError("`CausalBGM_IV_Image` requires `v_dim >= 785`.")
        if int(params.get("w_dim", 0)) != 1:
            raise ValueError("`CausalBGM_IV_Image` requires `w_dim == 1`.")

        super().__init__(params=params, timestamp=timestamp, random_seed=random_seed)

        z_dim = sum(self.params["z_dims"])
        self.image_dim = 28 * 28
        self.extra_noise_dim = int(self.params["v_dim"]) - 1 - self.image_dim
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

        self.ckpt = tf.train.Checkpoint(
            g_net=self.g_net,
            e_net=self.e_net,
            f_net=self.f_net,
            h_net=self.h_net,
            dz_net=self.dz_net,
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
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!!")

    def initialize_nets(self, print_summary=False):
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
        if print_summary:
            print(self.g_net.summary())
            print(self.e_net.summary())
            print(self.f_net.summary())
            print(self.h_net.summary())

    @staticmethod
    def _split_public_covariates(data_v):
        data_v = tf.cast(data_v, tf.float32)
        time = data_v[:, :1]
        image_raw = data_v[:, 1:785]
        noise = data_v[:, 785:]
        image_norm = image_raw / 255.0
        return time, image_raw, image_norm, noise

    def _decode_covariates(self, data_z, training=True):
        return self.g_net(data_z, training=training)

    def _covariate_loss_terms(self, data_v, data_z, training=True):
        time_obs, _, image_obs, noise_obs = self._split_public_covariates(data_v)
        decoded = self._decode_covariates(data_z, training=training)

        time_mean = decoded["time_mean"]
        time_var = decoded["time_var"]
        image_logits = decoded["image_logits"]
        image_probs = decoded["image_probs"]
        noise_mean = decoded["noise_mean"]
        noise_var = decoded["noise_var"]

        time_nll = tf.squeeze(
            ((time_obs - time_mean) ** 2) / (2.0 * time_var)
            + 0.5 * tf.math.log(time_var),
            axis=1,
        )
        image_nll = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=image_obs,
                logits=image_logits,
            ),
            axis=1,
        )
        mse_time = tf.reduce_mean((time_obs - time_mean) ** 2)
        mse_image = tf.reduce_mean((image_obs - image_probs) ** 2)
        loss_terms = [time_nll, image_nll]
        mse_terms = [mse_time, mse_image]
        if self.extra_noise_dim > 0:
            noise_nll = tf.reduce_mean(
                ((noise_obs - noise_mean) ** 2) / (2.0 * noise_var)
                + 0.5 * tf.math.log(noise_var),
                axis=1,
            )
            mse_noise = tf.reduce_mean((noise_obs - noise_mean) ** 2)
            loss_terms.append(noise_nll)
            mse_terms.append(mse_noise)
        mse_v = tf.add_n(mse_terms) / float(len(mse_terms))
        return tf.add_n(loss_terms), mse_v, decoded

    def _covariate_cycle_mse(self, observed_v, reconstructed_v):
        observed_time, _, observed_image, observed_noise = self._split_public_covariates(observed_v)
        reconstructed_time, _, reconstructed_image, reconstructed_noise = self._split_public_covariates(
            reconstructed_v
        )
        mse_time = tf.reduce_mean((observed_time - reconstructed_time) ** 2)
        mse_image = tf.reduce_mean((observed_image - reconstructed_image) ** 2)
        mse_terms = [mse_time, mse_image]
        if self.extra_noise_dim > 0:
            mse_terms.append(tf.reduce_mean((observed_noise - reconstructed_noise) ** 2))
        return tf.add_n(mse_terms) / float(len(mse_terms))

    @tf.function
    def update_g_net(self, data_z, data_v, eps=1e-6):
        del eps
        with tf.GradientTape() as gen_tape:
            loss_terms, loss_mse, _ = self._covariate_loss_terms(
                data_v, data_z, training=True
            )
            loss_v = tf.reduce_mean(loss_terms)
            if self.params["use_bnn"]:
                loss_v += sum(self.g_net.losses) * self.params["kl_weight"]

        g_gradients = gen_tape.gradient(loss_v, self.g_net.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g_net.trainable_variables))
        return loss_v, loss_mse

    @tf.function
    def update_latent_variable_sgd(
        self, data_x, data_y, data_v, data_w, batch_idx, include_outcome=True, eps=1e-6
    ):
        del eps
        with tf.GradientTape() as tape:
            data_z = tf.gather(self.data_z, batch_idx, axis=0)

            loss_pv_z, _, _ = self._covariate_loss_terms(
                data_v, data_z, training=True
            )
            loss_pv_z = tf.reduce_mean(loss_pv_z)

            treatment_output = self._treatment_output(data_z, data_w)
            mu_x = treatment_output[:, :1]
            if self.params["binary_treatment"]:
                loss_px_z = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, logits=mu_x)
                )
            else:
                sigma_square_x = self._continuous_sigma(
                    treatment_output, sigma_key="sigma_x"
                )
                loss_px_z = self._gaussian_nll(data_x, mu_x, sigma_square_x, event_dim=1)
                loss_px_z = tf.reduce_mean(loss_px_z)

            if include_outcome:
                loss_py_z = -tf.reduce_mean(
                    self._integrated_outcome_log_prob(
                        data_z,
                        data_w,
                        data_y,
                        n_samples=int(self.params["iv_mc_samples"]),
                    )
                )
            else:
                loss_py_z = tf.constant(0.0, dtype=tf.float32)

            loss_prior_z = tf.reduce_mean(tf.reduce_sum(data_z ** 2, axis=1) / 2.0)
            latent_pzv_weight = tf.cast(self.params["latent_pzv_weight"], tf.float32)
            loss_posterior_z = (
                latent_pzv_weight * (loss_pv_z + loss_prior_z)
                + loss_px_z
                + loss_py_z
            )

        posterior_gradients = tape.gradient(loss_posterior_z, [self.data_z])
        self.posterior_optimizer.apply_gradients(zip(posterior_gradients, [self.data_z]))
        return loss_posterior_z

    @tf.function
    def train_gen_step(self, data_z, data_v, data_w, data_x, data_y):
        with tf.GradientTape(persistent=True) as gen_tape:
            decoded = self._decode_covariates(data_z, training=True)
            data_v_ = decoded["public_v"]

            data_z_ = self.e_net(data_v, training=True)
            data_z0, data_z1, data_z2 = self._split_z(data_z_)
            data_z__ = self.e_net(data_v_, training=True)
            data_v__ = self._decode_covariates(data_z_, training=True)["public_v"]
            data_dz_ = self.dz_net(data_z_)

            l2_loss_v = self._covariate_cycle_mse(data_v, data_v__)
            l2_loss_z = tf.reduce_mean((data_z - data_z__) ** 2)
            e_loss_adv = -tf.reduce_mean(data_dz_)
            sigma_square_loss = tf.reduce_mean(tf.square(decoded["time_var"]))
            if self.extra_noise_dim > 0:
                sigma_square_loss += tf.reduce_mean(tf.square(decoded["noise_var"]))

            h_output = self.h_net(tf.concat([data_z0, data_z2, data_w], axis=-1))
            data_x_ = h_output[:, :1]
            if self.params["binary_treatment"]:
                l2_loss_x = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, logits=data_x_)
                )
                deconfounded_x = tf.sigmoid(data_x_)
            else:
                sigma_square_loss += tf.reduce_mean(tf.square(h_output[:, -1]))
                l2_loss_x = tf.reduce_mean((data_x_ - data_x) ** 2)
                deconfounded_x = data_x_

            f_output = self.f_net(tf.concat([data_z0, data_z1, deconfounded_x], axis=-1))
            data_y_ = f_output[:, :1]
            sigma_square_loss += tf.reduce_mean(tf.square(f_output[:, -1]))
            l2_loss_y = tf.reduce_mean((data_y_ - data_y) ** 2)

            g_e_loss = (
                e_loss_adv
                + (l2_loss_v + self.params["use_z_rec"] * l2_loss_z)
                + l2_loss_x
                + l2_loss_y
                + 0.001 * sigma_square_loss
            )

        trainable_variables = (
            self.g_net.trainable_variables
            + self.e_net.trainable_variables
            + self.f_net.trainable_variables
            + self.h_net.trainable_variables
        )
        g_e_gradients = gen_tape.gradient(g_e_loss, trainable_variables)
        self.g_pre_optimizer.apply_gradients(zip(g_e_gradients, trainable_variables))
        return e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss

    @tf.function
    def evaluate(self, data, data_z=None, nb_intervals=200):
        data_x, data_y, data_v, data_w = data
        if data_z is None:
            data_z = self.e_net(data_v, training=False)

        data_z0, data_z1, _ = self._split_z(data_z)
        data_v_pred = self._decode_covariates(data_z, training=False)["public_v"]
        data_x_pred = self._treatment_mean(data_z, data_w)
        data_y_pred = self._integrated_outcome_mean(
            data_z,
            data_w,
            n_samples=int(self.params["eval_mc_samples"]),
            sample_y=False,
        )

        mse_v = self._covariate_cycle_mse(data_v, data_v_pred)
        mse_x = tf.reduce_mean((data_x - data_x_pred) ** 2)
        mse_y = tf.reduce_mean((data_y - data_y_pred) ** 2)

        if self.params["binary_treatment"]:
            y_pred_pos = self.f_net(
                tf.concat([data_z0, data_z1, tf.ones((len(data_x), 1))], axis=-1)
            )[:, :1]
            y_pred_neg = self.f_net(
                tf.concat([data_z0, data_z1, tf.zeros((len(data_x), 1))], axis=-1)
            )[:, :1]
            ite_pre = y_pred_pos - y_pred_neg
            return ite_pre, mse_x, mse_y, mse_v

        x_min = tfp.stats.percentile(data_x, 5.0)
        x_max = tfp.stats.percentile(data_x, 95.0)
        x_values = tf.linspace(x_min, x_max, nb_intervals)

        def compute_dose_response(x):
            data_x_tile = tf.cast(tf.fill([tf.shape(data_x)[0], 1], x), tf.float32)
            y_pred = self.f_net(tf.concat([data_z0, data_z1, data_x_tile], axis=-1))[
                :, :1
            ]
            return tf.reduce_mean(y_pred)

        dose_response = tf.map_fn(
            compute_dose_response, x_values, fn_output_signature=tf.float32
        )
        return dose_response, mse_x, mse_y, mse_v

    @tf.function
    def get_log_posterior(self, data_x, data_y, data_v, data_w, data_z, eps=1e-6):
        del eps
        loss_pv_z, _, _ = self._covariate_loss_terms(data_v, data_z, training=False)

        treatment_output = self._treatment_output(data_z, data_w)
        mu_x = treatment_output[:, :1]
        if self.params["binary_treatment"]:
            loss_px_z = tf.squeeze(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, logits=mu_x)
            )
        else:
            sigma_square_x = self._continuous_sigma(treatment_output, sigma_key="sigma_x")
            loss_px_z = self._gaussian_nll(data_x, mu_x, sigma_square_x, event_dim=1)

        loss_py_z = -self._integrated_outcome_log_prob(
            data_z,
            data_w,
            data_y,
            n_samples=int(self.params["iv_mc_samples"]),
        )
        loss_prior_z = tf.reduce_sum(data_z ** 2, axis=1) / 2.0
        return -(loss_pv_z + loss_px_z + loss_py_z + loss_prior_z)

    @tf.function
    def get_log_partial_posterior(self, data_x, data_v, data_w, data_z, eps=1e-6):
        del eps
        loss_pv_z, _, _ = self._covariate_loss_terms(data_v, data_z, training=False)

        treatment_output = self._treatment_output(data_z, data_w)
        mu_x = treatment_output[:, :1]
        if self.params["binary_treatment"]:
            loss_px_z = tf.squeeze(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, logits=mu_x)
            )
        else:
            sigma_square_x = self._continuous_sigma(treatment_output, sigma_key="sigma_x")
            loss_px_z = self._gaussian_nll(data_x, mu_x, sigma_square_x, event_dim=1)

        loss_prior_z = tf.reduce_sum(data_z ** 2, axis=1) / 2.0
        return -(loss_pv_z + loss_px_z + loss_prior_z)

    @tf.function
    def get_log_covariate_posterior(self, data_v, data_z, eps=1e-6):
        del eps
        loss_pv_z, _, _ = self._covariate_loss_terms(data_v, data_z, training=False)
        loss_prior_z = tf.reduce_sum(data_z ** 2, axis=1) / 2.0
        return -(loss_pv_z + loss_prior_z)
