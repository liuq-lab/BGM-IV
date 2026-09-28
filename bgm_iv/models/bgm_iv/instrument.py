import datetime
import os

import dateutil.tz
import numpy as np
import tensorflow as tf

from ..networks import BaseFullyConnectedNet, Discriminator

_EGM_GH_NODES = 8
_EGM_LOG_EVERY = 500
_SIGMA_Y_FLOOR = 0.1


class BGM_IV:
    """Causally structured Bayesian generative model for IV regression."""

    def __init__(
        self,
        params,
        timestamp=None,
        random_seed=None,
        auto_restore_checkpoint=True,
    ):
        if "w_dim" not in params:
            raise KeyError("`w_dim` must be provided in params for BGM_IV.")

        self.params = dict(params)
        if int(self.params["structural_map_steps"]) < 1:
            raise ValueError("`structural_map_steps` must be at least 1")
        self.timestamp = timestamp

        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            tf.config.experimental.enable_op_determinism()

        z_dim = sum(self.params["z_dims"])
        z0_dim = self.params["z_dims"][0]
        z1_dim = self.params["z_dims"][1]
        z2_dim = self.params["z_dims"][2]

        self.g_net = BaseFullyConnectedNet(
            input_dim=z_dim,
            output_dim=self.params["v_dim"] + 1,
            model_name="g_net",
            nb_units=self.params["g_units"],
        )
        self.e_net = BaseFullyConnectedNet(
            input_dim=self.params["v_dim"],
            output_dim=z_dim,
            model_name="e_net",
            nb_units=self.params["e_units"],
        )
        self.f_net = BaseFullyConnectedNet(
            input_dim=z0_dim + z1_dim + 1,
            output_dim=2,
            model_name="f_net",
            nb_units=self.params["f_units"],
        )
        self.h_net = BaseFullyConnectedNet(
            input_dim=z0_dim + z2_dim + self.params["w_dim"],
            output_dim=2,
            model_name="h_net",
            nb_units=self.params["h_units"],
        )

        self.dz_net = Discriminator(
            input_dim=z_dim, model_name="dz_net", nb_units=self.params["dz_units"]
        )

        self.g_pre_optimizer = tf.keras.optimizers.Adam(
            self.params["lr"], beta_1=0.9, beta_2=0.99
        )
        self.d_pre_optimizer = tf.keras.optimizers.Adam(
            self.params["lr"], beta_1=0.9, beta_2=0.99
        )
        gh_t, gh_w = np.polynomial.hermite.hermgauss(_EGM_GH_NODES)
        self._egm_gh_t = tf.constant(gh_t.reshape(_EGM_GH_NODES, 1, 1), dtype=tf.float32)
        self._egm_gh_w = tf.constant(
            (gh_w / np.sqrt(np.pi)).reshape(_EGM_GH_NODES, 1, 1), dtype=tf.float32
        )
        self.egm_sigma2_x_ema = tf.Variable(
            1.0, trainable=False, dtype=tf.float32, name="egm_sigma2_x_ema"
        )

        self.g_optimizer = tf.keras.optimizers.Adam(
            self.params["lr_theta"], beta_1=0.9, beta_2=0.99
        )
        self.f_optimizer = tf.keras.optimizers.Adam(
            self.params["lr_theta"], beta_1=0.9, beta_2=0.99
        )
        self.h_optimizer = tf.keras.optimizers.Adam(
            self.params["lr_theta"], beta_1=0.9, beta_2=0.99
        )
        self.posterior_optimizer = tf.keras.optimizers.Adam(
            self.params["lr_z"], beta_1=0.9, beta_2=0.99
        )

        self.initialize_nets()

        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime("%Y%m%d_%H%M%S_%f")

        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            self.params["output_dir"], self.params["dataset"], self.timestamp
        )
        if self.params.get("save_model", False) and not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

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

    def make_model_state_checkpoint(self):
        return tf.train.Checkpoint(
            g_net=self.g_net,
            e_net=self.e_net,
            f_net=self.f_net,
            h_net=self.h_net,
            dz_net=self.dz_net,
            egm_sigma2_x_ema=self.egm_sigma2_x_ema,
        )

    def save_model_state_checkpoint(self, checkpoint_prefix):
        checkpoint_prefix = os.fspath(checkpoint_prefix)
        directory = os.path.dirname(checkpoint_prefix)
        if directory:
            os.makedirs(directory, exist_ok=True)
        checkpoint = self.make_model_state_checkpoint()
        return checkpoint.save(checkpoint_prefix)

    def restore_model_state_checkpoint(self, checkpoint_prefix):
        checkpoint = self.make_model_state_checkpoint()
        status = checkpoint.restore(os.fspath(checkpoint_prefix))
        status.assert_consumed()
        return status

    @staticmethod
    def _to_2d_float32(array):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array

    def _parse_train_data(self, data):
        if len(data) != 4:
            raise ValueError(
                "BGM_IV.fit/evaluate expect data=(x, y, v, w)."
            )
        data_x, data_y, data_v, data_w = data
        data_x = self._to_2d_float32(data_x)
        data_y = self._to_2d_float32(data_y)
        data_v = self._to_2d_float32(data_v)
        data_w = self._to_2d_float32(data_w)
        if data_v.shape[1] != self.params["v_dim"]:
            raise ValueError(
                f"`v` has dim {data_v.shape[1]}, expected {self.params['v_dim']}."
            )
        if data_w.shape[1] != self.params["w_dim"]:
            raise ValueError(
                f"`w` has dim {data_w.shape[1]}, expected {self.params['w_dim']}."
            )
        return data_x, data_y, data_v, data_w

    def initialize_nets(self):
        z_dim = sum(self.params["z_dims"])
        z0_dim = self.params["z_dims"][0]
        z1_dim = self.params["z_dims"][1]
        z2_dim = self.params["z_dims"][2]
        self.g_net(np.zeros((1, z_dim), dtype=np.float32))
        self.e_net(np.zeros((1, self.params["v_dim"]), dtype=np.float32))
        self.f_net(np.zeros((1, z0_dim + z1_dim + 1), dtype=np.float32))
        self.h_net(
            np.zeros(
                (1, z0_dim + z2_dim + self.params["w_dim"]), dtype=np.float32
            )
        )

    def _split_z(self, data_z):
        z0_dim = self.params["z_dims"][0]
        z1_dim = self.params["z_dims"][1]
        z2_dim = self.params["z_dims"][2]
        data_z0 = data_z[:, :z0_dim]
        data_z1 = data_z[:, z0_dim : z0_dim + z1_dim]
        data_z2 = data_z[:, z0_dim + z1_dim : z0_dim + z1_dim + z2_dim]
        return data_z0, data_z1, data_z2

    def _treatment_output(self, data_z, data_w):
        data_z0, _, data_z2 = self._split_z(data_z)
        return self.h_net(tf.concat([data_z0, data_z2, data_w], axis=-1))

    def _outcome_output(self, data_z, data_x):
        data_z0, data_z1, _ = self._split_z(data_z)
        return self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))

    def _continuous_sigma(self, net_output, sigma_key, eps=1e-6):
        learned_var = tf.nn.softplus(net_output[:, -1:]) + eps
        if sigma_key != "sigma_y":
            return learned_var
        sigma_min = tf.cast(_SIGMA_Y_FLOOR, tf.float32)
        return tf.square(sigma_min + tf.sqrt(learned_var))

    def _gaussian_nll(self, targets, means, sigma_square, event_dim):
        sq_error = tf.reduce_sum((targets - means) ** 2, axis=1, keepdims=True)
        nll = sq_error / (2.0 * sigma_square) + 0.5 * event_dim * tf.math.log(
            sigma_square
        )
        return tf.squeeze(nll, axis=1)

    def _covariate_nll(self, data_v, data_z, training, eps=1e-6):
        del training
        g_output = self.g_net(data_z)
        mu_v = g_output[:, : self.params["v_dim"]]
        sigma_square_v = self._continuous_sigma(g_output, sigma_key="sigma_v", eps=eps)
        return self._gaussian_nll(
            data_v,
            mu_v,
            sigma_square_v,
            event_dim=self.params["v_dim"],
        )

    def _covariate_train_loss(self, data_z, data_v, eps=1e-6):
        g_net_output = self.g_net(data_z)
        mu_v = g_net_output[:, : self.params["v_dim"]]
        sigma_square_v = tf.nn.softplus(g_net_output[:, -1]) + eps
        loss_v = tf.reduce_sum((data_v - mu_v) ** 2, axis=1) / (
            2 * sigma_square_v
        ) + self.params["v_dim"] * tf.math.log(sigma_square_v) / 2
        return tf.reduce_mean(loss_v)

    def _covariate_reconstruction(self, data_z):
        return self.g_net(data_z)[:, : self.params["v_dim"]]

    def _covariate_cycle_mse(self, observed_v, reconstructed_v):
        return tf.reduce_mean((observed_v - reconstructed_v) ** 2)

    def _egm_covariate_block(self, data_z, data_v):
        sigma_square_loss = 0.0
        g_output = self.g_net(data_z)
        data_v_ = g_output[:, : self.params["v_dim"]]
        sigma_square_loss += tf.reduce_mean(tf.square(g_output[:, -1]))

        data_z_ = self.e_net(data_v)
        data_z0, data_z1, data_z2 = self._split_z(data_z_)

        data_z__ = self.e_net(data_v_)
        data_v__ = self.g_net(data_z_)[:, : self.params["v_dim"]]
        return sigma_square_loss, data_z_, (data_z0, data_z1, data_z2), data_z__, data_v__

    def _treatment_mean(self, data_z, data_w):
        treatment_output = self._treatment_output(data_z, data_w)
        return treatment_output[:, :1]

    @tf.function
    def _covariate_neg_log_posterior(self, data_v, data_z, eps=1e-6):
        return -tf.reduce_mean(self.get_log_covariate_posterior(data_v, data_z, eps=eps))

    def encoder_latent(self, data_v):
        data_v = self._to_2d_float32(data_v)
        return self.e_net(data_v, training=False).numpy().astype(np.float32)

    def infer_latent_from_covariates(self, data_v, map_steps=None, map_lr=None):
        data_v = self._to_2d_float32(data_v)
        initial_z = self.encoder_latent(data_v)

        if map_steps is None:
            map_steps = int(self.params["structural_map_steps"])
        if map_lr is None:
            map_lr = float(self.params["structural_map_lr"])

        data_v_tf = tf.convert_to_tensor(data_v, dtype=tf.float32)
        data_z = tf.Variable(initial_z.astype(np.float32), trainable=True)
        optimizer = tf.keras.optimizers.Adam(map_lr, beta_1=0.9, beta_2=0.99)

        for _ in range(map_steps):
            with tf.GradientTape() as tape:
                loss = self._covariate_neg_log_posterior(data_v_tf, data_z)
            gradients = tape.gradient(loss, [data_z])
            optimizer.apply_gradients(zip(gradients, [data_z]))

        return data_z.numpy().astype(np.float32)

    def _sample_treatment(self, data_z, data_w, n_samples=1, eps=1e-6):
        treatment_output = self._treatment_output(data_z, data_w)
        mu_x = treatment_output[:, :1]
        sigma_square_x = self._continuous_sigma(
            treatment_output, sigma_key="sigma_x", eps=eps
        )
        eps_x = tf.random.normal(
            [n_samples, tf.shape(mu_x)[0], 1], dtype=mu_x.dtype
        )
        return mu_x[None, :, :] + eps_x * tf.sqrt(sigma_square_x)[None, :, :]

    def _outcome_outputs_for_samples(self, data_z, x_samples, training=True):
        z0_dim = self.params["z_dims"][0]
        z1_dim = self.params["z_dims"][1]
        data_z0, data_z1, _ = self._split_z(data_z)
        n_samples = tf.shape(x_samples)[0]
        n_obs = tf.shape(data_z)[0]

        data_z0_tiled = tf.tile(tf.expand_dims(data_z0, axis=0), [n_samples, 1, 1])
        data_z1_tiled = tf.tile(tf.expand_dims(data_z1, axis=0), [n_samples, 1, 1])
        flat_inputs = tf.concat(
            [
                tf.reshape(data_z0_tiled, (-1, z0_dim)),
                tf.reshape(data_z1_tiled, (-1, z1_dim)),
                tf.reshape(x_samples, (-1, 1)),
            ],
            axis=-1,
        )
        flat_outputs = self.f_net(flat_inputs, training=training)
        return tf.reshape(flat_outputs, (n_samples, n_obs, 2))

    def _integrated_outcome_log_prob(self, data_z, data_w, data_y, n_samples=None, eps=1e-6):
        if n_samples is None:
            n_samples = int(self.params["iv_mc_samples"])
        x_samples = self._sample_treatment(data_z, data_w, n_samples=n_samples, eps=eps)
        outcome_outputs = self._outcome_outputs_for_samples(data_z, x_samples)
        mu_y = outcome_outputs[:, :, :1]
        sigma_square_y = self._continuous_sigma(
            tf.reshape(outcome_outputs, (-1, 2)), sigma_key="sigma_y", eps=eps
        )
        sigma_square_y = tf.reshape(sigma_square_y, tf.shape(mu_y))
        data_y = tf.expand_dims(data_y, axis=0)
        log_prob_samples = -(
            (data_y - mu_y) ** 2 / (2.0 * sigma_square_y)
            + 0.5 * tf.math.log(sigma_square_y)
        )
        log_prob_samples = tf.squeeze(log_prob_samples, axis=-1)
        return tf.reduce_logsumexp(log_prob_samples, axis=0) - tf.math.log(
            tf.cast(n_samples, tf.float32)
        )

    def _integrated_outcome_mean(self, data_z, data_w, n_samples=None, eps=1e-6):
        if n_samples is None:
            n_samples = int(self.params["eval_mc_samples"])
        x_samples = self._sample_treatment(data_z, data_w, n_samples=n_samples, eps=eps)
        outcome_outputs = self._outcome_outputs_for_samples(data_z, x_samples)
        return tf.reduce_mean(outcome_outputs[:, :, :1], axis=0)

    @tf.function
    def update_g_net(self, data_z, data_v, eps=1e-6):
        with tf.GradientTape() as gen_tape:
            loss_v = self._covariate_train_loss(data_z, data_v, eps=eps)

        g_gradients = gen_tape.gradient(loss_v, self.g_net.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g_net.trainable_variables))
        return loss_v

    @tf.function
    def update_h_net(self, data_z, data_w, data_x, eps=1e-6):
        with tf.GradientTape() as gen_tape:
            treatment_output = self._treatment_output(data_z, data_w)
            mu_x = treatment_output[:, :1]
            sigma_square_x = self._continuous_sigma(
                treatment_output, sigma_key="sigma_x", eps=eps
            )
            loss_x = self._gaussian_nll(data_x, mu_x, sigma_square_x, event_dim=1)
            loss_x = tf.reduce_mean(loss_x)

        h_gradients = gen_tape.gradient(loss_x, self.h_net.trainable_variables)
        self.h_optimizer.apply_gradients(zip(h_gradients, self.h_net.trainable_variables))
        return loss_x

    @tf.function
    def update_f_net(self, data_z, data_w, data_y, n_samples=None, eps=1e-6):
        if n_samples is None:
            n_samples = int(self.params["iv_mc_samples"])

        with tf.GradientTape() as gen_tape:
            log_prob = self._integrated_outcome_log_prob(
                data_z, data_w, data_y, n_samples=n_samples, eps=eps
            )
            loss_y = -tf.reduce_mean(log_prob)

        f_gradients = gen_tape.gradient(loss_y, self.f_net.trainable_variables)
        self.f_optimizer.apply_gradients(zip(f_gradients, self.f_net.trainable_variables))
        return loss_y

    @tf.function
    def update_latent_variable_sgd(
        self, data_x, data_y, data_v, data_w, batch_idx, include_outcome=True, eps=1e-6
    ):
        with tf.GradientTape() as tape:
            data_z = tf.gather(self.data_z, batch_idx, axis=0)

            loss_pv_z = tf.reduce_mean(
                self._covariate_nll(data_v, data_z, training=True, eps=eps)
            )

            treatment_output = self._treatment_output(data_z, data_w)
            mu_x = treatment_output[:, :1]
            sigma_square_x = self._continuous_sigma(
                treatment_output, sigma_key="sigma_x", eps=eps
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
                        eps=eps,
                    )
                )
            else:
                loss_py_z = tf.constant(0.0, dtype=tf.float32)

            loss_prior_z = tf.reduce_mean(tf.reduce_sum(data_z ** 2, axis=1) / 2.0)
            loss_posterior_z = (
                loss_pv_z + loss_prior_z + loss_px_z
                + self._outcome_to_particles_weight() * loss_py_z
            )

        posterior_gradients = tape.gradient(loss_posterior_z, [self.data_z])
        self.posterior_optimizer.apply_gradients(
            zip(posterior_gradients, [self.data_z])
        )
        return loss_posterior_z

    @tf.function
    def train_disc_step(self, data_z, data_v):
        epsilon_z = tf.random.uniform([], minval=0.0, maxval=1.0)
        with tf.GradientTape(persistent=True) as disc_tape:
            with tf.GradientTape() as gp_tape:
                data_z_ = self.e_net(data_v)
                data_z_hat = data_z * epsilon_z + data_z_ * (1 - epsilon_z)
                data_dz_hat = self.dz_net(data_z_hat)

            data_dz_ = self.dz_net(data_z_)
            data_dz = self.dz_net(data_z)
            dz_loss = -tf.reduce_mean(data_dz) + tf.reduce_mean(data_dz_)

            grad_z = gp_tape.gradient(data_dz_hat, data_z_hat)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))

            d_loss = dz_loss + 10 * gpz_loss

        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables)
        self.d_pre_optimizer.apply_gradients(
            zip(d_gradients, self.dz_net.trainable_variables)
        )
        return dz_loss, d_loss

    @tf.function
    def train_gen_step(self, data_z, data_v, data_w, data_x, data_y):
        with tf.GradientTape(persistent=True) as gen_tape:
            (
                sigma_square_loss,
                data_z_,
                (data_z0, data_z1, data_z2),
                data_z__,
                data_v__,
            ) = self._egm_covariate_block(data_z, data_v)
            data_dz_ = self.dz_net(data_z_)

            l2_loss_v = self._covariate_cycle_mse(data_v, data_v__)
            l2_loss_z = tf.reduce_mean((data_z - data_z__) ** 2)
            e_loss_adv = -tf.reduce_mean(data_dz_)

            h_output = self.h_net(tf.concat([data_z0, data_z2, data_w], axis=-1))
            data_x_ = h_output[:, :1]
            sigma_square_loss += tf.reduce_mean(tf.square(h_output[:, -1]))
            l2_loss_x = tf.reduce_mean((data_x_ - data_x) ** 2)

            batch_resid = tf.stop_gradient(
                tf.reduce_mean(tf.square(data_x - data_x_))
            )
            self.egm_sigma2_x_ema.assign(
                0.99 * self.egm_sigma2_x_ema + 0.01 * batch_resid
            )
            sigma_square_x = self.egm_sigma2_x_ema * tf.ones_like(data_x_)
            sigma_square_x = tf.minimum(sigma_square_x, 1.0)

            # Stop-gradient sigma: f cannot lower the loss by widening the integral.
            sigma_x = tf.stop_gradient(tf.sqrt(sigma_square_x))
            x_nodes = (
                data_x_[None, :, :]
                + 1.4142135623730951 * sigma_x[None, :, :] * self._egm_gh_t
            )
            f_outputs = self._outcome_outputs_for_samples(data_z_, x_nodes)
            data_y_ = tf.reduce_sum(self._egm_gh_w * f_outputs[:, :, :1], axis=0)
            sigma_square_loss += tf.reduce_mean(tf.square(f_outputs[:, :, -1]))

            l2_loss_y = tf.reduce_mean((data_y_ - data_y) ** 2)

            g_e_loss = (
                e_loss_adv
                + (l2_loss_v + l2_loss_z)
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

    def egm_init(
        self,
        data,
        egm_n_iter=30000,
        batch_size=32,
        verbose=1,
    ):
        data_x, data_y, data_v, data_w = self._parse_train_data(data)

        print("EGM Initialization Starts ...")
        z_dim = sum(self.params["z_dims"])
        for batch_iter in range(egm_n_iter + 1):
            for _ in range(self.params["g_d_freq"]):
                batch_idx = np.random.choice(len(data_x), batch_size, replace=False)
                batch_z = np.random.normal(0.0, 1.0, (batch_size, z_dim)).astype("float32")
                batch_v = data_v[batch_idx, :]
                dz_loss, d_loss = self.train_disc_step(batch_z, batch_v)

            batch_z = np.random.normal(0.0, 1.0, (batch_size, z_dim)).astype("float32")
            batch_idx = np.random.choice(len(data_x), batch_size, replace=False)
            batch_x = data_x[batch_idx, :]
            batch_y = data_y[batch_idx, :]
            batch_v = data_v[batch_idx, :]
            batch_w = data_w[batch_idx, :]
            e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss = (
                self.train_gen_step(batch_z, batch_v, batch_w, batch_x, batch_y)
            )
            if batch_iter % _EGM_LOG_EVERY == 0:
                loss_contents = (
                    "EGM Initialization Iter [%d] : e_loss_adv [%.4f], l2_loss_v [%.4f], "
                    "l2_loss_z [%.4f], l2_loss_x [%.4f], l2_loss_y [%.4f], g_e_loss [%.4f], "
                    "dz_loss [%.4f], d_loss [%.4f]"
                    % (
                        batch_iter,
                        e_loss_adv,
                        l2_loss_v,
                        l2_loss_z,
                        l2_loss_x,
                        l2_loss_y,
                        g_e_loss,
                        dz_loss,
                        d_loss,
                    )
                )
                if verbose:
                    print(loss_contents)
        print("EGM Initialization Ends.")

    def _apply_bn_determinism(self):
        for attr in vars(self).values():
            if isinstance(attr, tf.Module):
                for layer in getattr(attr, "submodules", ()):
                    if isinstance(layer, tf.keras.layers.BatchNormalization):
                        layer.fused = False

    def _outcome_to_particles_weight(self):
        gamma = float(self.params.get("outcome_to_particles_weight", 0.01))
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("outcome_to_particles_weight must be in [0, 1]")
        return gamma

    def fit(
        self,
        data,
        epochs=100,
        epochs_per_eval=5,
        batch_size=32,
        startoff=0,
        use_egm_init=True,
        egm_n_iter=30000,
        verbose=1,
    ):
        self._apply_bn_determinism()
        self._parse_train_data(data)

        if use_egm_init:
            self.egm_init(
                data,
                egm_n_iter=egm_n_iter,
                batch_size=batch_size,
                verbose=verbose,
            )
        self.fit_bgm_from_egm(
            data,
            epochs=epochs,
            epochs_per_eval=epochs_per_eval,
            batch_size=batch_size,
            startoff=startoff,
            verbose=verbose,
            initialize_latents_from_encoder=use_egm_init,
        )

    def fit_bgm_from_egm(
        self,
        data,
        epochs=100,
        epochs_per_eval=5,
        batch_size=32,
        startoff=0,
        verbose=1,
        initialize_latents_from_encoder=True,
    ):
        self._apply_bn_determinism()
        data_x, data_y, data_v, data_w = self._parse_train_data(data)

        if initialize_latents_from_encoder:
            print("Initialize latent variables Z with e(V)...")
            data_z_init = self.e_net(data_v)
        else:
            print("Random initialization of latent variables Z...")
            data_z_init = np.random.normal(
                0,
                1,
                size=(len(data_x), sum(self.params["z_dims"])),
            ).astype("float32")

        self.data_z = tf.Variable(data_z_init, name="Latent Variable", trainable=True)
        self.ckpt.data_z = self.data_z

        print("Iterative Updating Starts ...")
        for epoch in range(epochs + 1):
            sample_idx = np.random.choice(len(data_x), len(data_x), replace=False)
            gamma = self._outcome_to_particles_weight()
            particle_outcome = gamma > 0.0

            for i in range(0, len(data_x), batch_size):
                batch_idx = sample_idx[i : i + batch_size]
                batch_z = tf.gather(self.data_z, batch_idx, axis=0)
                batch_x = data_x[batch_idx, :]
                batch_y = data_y[batch_idx, :]
                batch_v = data_v[batch_idx, :]
                batch_w = data_w[batch_idx, :]

                self.update_g_net(batch_z, batch_v)
                self.update_h_net(batch_z, batch_w, batch_x)
                self.update_f_net(
                    batch_z,
                    batch_w,
                    batch_y,
                    n_samples=int(self.params["iv_mc_samples"]),
                )
                self.update_latent_variable_sgd(
                    batch_x,
                    batch_y,
                    batch_v,
                    batch_w,
                    batch_idx,
                    include_outcome=particle_outcome,
                )

            if epoch % epochs_per_eval == 0:
                if verbose:
                    print("Epoch [%d/%d]" % (epoch, epochs))
                if epoch >= startoff:
                    if self.params.get("save_model", False):
                        ckpt_save_path = self.ckpt_manager.save(epoch)
                        print(
                            "Saving checkpoint for epoch {} at {}".format(
                                epoch, ckpt_save_path
                            )
                        )

    @tf.function
    def evaluate(self, data, data_z=None):
        data_x, data_y, data_v, data_w = data
        if data_z is None:
            data_z = self.e_net(data_v, training=False)

        data_v_pred = self._covariate_reconstruction(data_z)
        data_x_pred = self._treatment_mean(data_z, data_w)
        data_y_pred = self._integrated_outcome_mean(
            data_z,
            data_w,
            n_samples=int(self.params["eval_mc_samples"]),
        )

        mse_v = self._covariate_cycle_mse(data_v, data_v_pred)
        mse_x = tf.reduce_mean((data_x - data_x_pred) ** 2)
        mse_y = tf.reduce_mean((data_y - data_y_pred) ** 2)
        return mse_x, mse_y, mse_v

    def predict_structural(self, data_x, data_v, map_steps=None, map_lr=None):
        data_x = self._to_2d_float32(data_x)
        data_v = self._to_2d_float32(data_v)
        data_z = self.infer_latent_from_covariates(
            data_v, map_steps=map_steps, map_lr=map_lr
        )
        data_z = tf.convert_to_tensor(data_z, dtype=tf.float32)
        data_z0, data_z1, _ = self._split_z(data_z)
        outcome_output = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))
        return outcome_output[:, :1].numpy()

    @tf.function
    def get_log_covariate_posterior(self, data_v, data_z, eps=1e-6):
        loss_pv_z = self._covariate_nll(data_v, data_z, training=False, eps=eps)
        loss_prior_z = tf.reduce_sum(data_z ** 2, axis=1) / 2.0
        return -(loss_pv_z + loss_prior_z)
