import datetime
import os

import dateutil.tz
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from bgm_iv.datasets import Gaussian_sampler
from bgm_iv.utils.data_io import save_data

from ..networks import BaseFullyConnectedNet, BayesianFullyConnectedNet, Discriminator
from .base import CausalBGM


class BGM_IV(CausalBGM):
    """Instrumental-variable extension of :class:`CausalBGM`.

    The observed tuple is ``(X, Y, V, W)`` where ``W`` denotes instrumental
    variables. The model treats ``W`` as fixed observed variables and learns

    - ``p(V | Z)``
    - ``p(X | W, Z0, Z2)``
    - ``p(Y | x, Z0, Z1)``

    The outcome likelihood is replaced by the IV pseudo-likelihood

    ``p(Y | W, Z) = integral p(Y | x, Z0, Z1) p(x | W, Z0, Z2) dx``

    which is evaluated by Monte Carlo for continuous treatments and exactly for
    binary treatments.
    """

    def __init__(self, params, timestamp=None, random_seed=None):
        if "w_dim" not in params:
            raise KeyError("`w_dim` must be provided in params for BGM_IV.")
        if "alpha_v" in params:
            raise ValueError(
                "`alpha_v` is no longer supported; test-time MAP uses the "
                "untempered covariate posterior."
            )

        self.params = dict(params)
        self.params.setdefault("iv_mc_samples", 16)
        self.params.setdefault("eval_mc_samples", self.params["iv_mc_samples"])
        self.params.setdefault("first_stage_warmup_epochs", 0)
        self.params.setdefault("structural_latent_method", "map")
        self.params.setdefault("structural_map_steps", 100)
        self.params.setdefault("structural_map_lr", self.params["lr_z"])
        self.params.setdefault("egm_outcome_loss", "integral")
        self.params.setdefault("egm_outcome_gh_nodes", 8)
        self.params.setdefault("egm_outcome_sigma", "residual_ema") 
        self.params.setdefault("egm_outcome_sigma_cap", 1.0)
        self.params.setdefault("egm_outcome_grad_path", "mean")  # "mean" | "stop"
        # `mcmc_seed` seeds every test-time stochastic procedure.
        self.params.setdefault("mcmc_seed", None)
        configured_scale = str(self.params.get("covariate_block_scale", "sum"))
        if configured_scale != "sum":
            raise ValueError("covariate blocks are fixed to event-sum reduction")
        self.params["covariate_block_scale"] = "sum"
        self.timestamp = timestamp

        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            tf.config.experimental.enable_op_determinism()

        z_dim = sum(self.params["z_dims"])
        z0_dim = self.params["z_dims"][0]
        z1_dim = self.params["z_dims"][1]
        z2_dim = self.params["z_dims"][2]

        if self.params["use_bnn"]:
            net_cls = BayesianFullyConnectedNet
        else:
            net_cls = BaseFullyConnectedNet

        self.g_net = net_cls(
            input_dim=z_dim,
            output_dim=self.params["v_dim"] + 1,
            model_name="g_net",
            nb_units=self.params["g_units"],
        )
        self.e_net = net_cls(
            input_dim=self.params["v_dim"],
            output_dim=z_dim,
            model_name="e_net",
            nb_units=self.params["e_units"],
        )
        self.f_net = net_cls(
            input_dim=z0_dim + z1_dim + 1,
            output_dim=2,
            model_name="f_net",
            nb_units=self.params["f_units"],
        )
        self.h_net = net_cls(
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
        self.z_sampler = Gaussian_sampler(mean=np.zeros(z_dim), sd=1.0)
        gh_nodes = int(self.params["egm_outcome_gh_nodes"])
        gh_t, gh_w = np.polynomial.hermite.hermgauss(gh_nodes)
        self._egm_gh_t = tf.constant(gh_t.reshape(gh_nodes, 1, 1), dtype=tf.float32)
        self._egm_gh_w = tf.constant(
            (gh_w / np.sqrt(np.pi)).reshape(gh_nodes, 1, 1), dtype=tf.float32
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
            self.timestamp = now.strftime("%Y%m%d_%H%M%S")

        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            self.params["output_dir"], self.params["dataset"], self.timestamp
        )
        if self.params["save_model"] and not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.save_dir = "{}/results/{}/{}".format(
            self.params["output_dir"], self.params["dataset"], self.timestamp
        )
        if self.params["save_res"] and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

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

    def _parse_predict_data(self, data):
        if len(data) == 3:
            data_x, data_v, data_w = data
            data_y = None
        elif len(data) == 4:
            data_x, data_y, data_v, data_w = data
            data_y = self._to_2d_float32(data_y)
        else:
            raise ValueError(
                "BGM_IV.predict expects data=(x, v, w) or data=(x, y, v, w)."
            )
        data_x = self._to_2d_float32(data_x)
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

    def _prepare_target_x(self, reference_x, x_values=None):
        if x_values is None:
            return self._to_2d_float32(reference_x)
        if np.isscalar(x_values):
            return np.full_like(reference_x, float(x_values), dtype=np.float32)

        target_x = self._to_2d_float32(x_values)
        if target_x.shape[0] == 1 and reference_x.shape[0] > 1:
            target_x = np.repeat(target_x, reference_x.shape[0], axis=0)
        if target_x.shape[0] != reference_x.shape[0]:
            raise ValueError(
                "`x_values` must be a scalar or have the same number of rows as `x`."
            )
        return target_x.astype(np.float32)

    def _coerce_metric_value(self, value):
        if isinstance(value, tf.Tensor):
            value = value.numpy()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return float(value)
            return value.tolist()
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        return value

    def _record_training_metrics(
        self,
        stage,
        epoch,
        mse_x,
        mse_y,
        mse_v,
        evaluation_callback=None,
        include_outcome=True,
    ):
        """Append a training-history row and enrich it via an optional callback."""
        record = {
            "stage": stage,
            "epoch": None if epoch is None else int(epoch),
            "include_outcome": bool(include_outcome),
            "mse_x": float(mse_x),
            "mse_y": float(mse_y),
            "mse_v": float(mse_v),
        }
        if evaluation_callback is not None:
            extra_metrics = evaluation_callback(
                model=self,
                stage=stage,
                epoch=epoch,
                metrics=dict(record),
            )
            if extra_metrics:
                for key, value in extra_metrics.items():
                    record[key] = self._coerce_metric_value(value)
        self.training_history.append(record)
        return record

    def _format_training_record(self, record):
        pieces = [
            f"MSE_x={record['mse_x']:.4f}",
            f"MSE_y={record['mse_y']:.4f}",
            f"MSE_v={record['mse_v']:.4f}",
        ]
        if "structural_mse" in record:
            pieces.append(f"structural_MSE={record['structural_mse']:.4f}")
        return ", ".join(pieces)

    def initialize_nets(self, print_summary=False):
        """Build all neural networks once so TensorFlow variables exist.

        Parameters
        ----------
        print_summary : bool, default=False
            If ``True``, print Keras summaries for the main sub-networks.
        """
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
        if print_summary:
            print(self.g_net.summary())
            print(self.e_net.summary())
            print(self.f_net.summary())
            print(self.h_net.summary())

    def _split_z(self, data_z):
        z0_dim = self.params["z_dims"][0]
        z1_dim = self.params["z_dims"][1]
        z2_dim = self.params["z_dims"][2]
        data_z0 = data_z[:, :z0_dim]
        data_z1 = data_z[:, z0_dim : z0_dim + z1_dim]
        data_z2 = data_z[:, z0_dim + z1_dim : z0_dim + z1_dim + z2_dim]
        return data_z0, data_z1, data_z2

    def _treatment_output(self, data_z, data_w):
        """Evaluate the treatment network.

        Parameters
        ----------
        data_z : tf.Tensor
            Latent variables with shape ``(n, sum(z_dims))``.
        data_w : tf.Tensor
            Instruments with shape ``(n, w_dim)``.

        Returns
        -------
        tf.Tensor
            Raw treatment-network output with shape ``(n, 2)`` where the
            first column is the treatment mean/logit and the second column
            parameterises the variance for continuous treatment.
        """
        data_z0, _, data_z2 = self._split_z(data_z)
        return self.h_net(tf.concat([data_z0, data_z2, data_w], axis=-1))

    def _outcome_output(self, data_z, data_x):
        """Evaluate the structural outcome network ``f(z0, z1, x)``.

        Parameters
        ----------
        data_z : tf.Tensor
            Latent variables with shape ``(n, sum(z_dims))``.
        data_x : tf.Tensor
            Treatments with shape ``(n, 1)``.

        Returns
        -------
        tf.Tensor
            Outcome-network output with shape ``(n, 2)`` where the first
            column is the conditional mean and the second column
            parameterises the variance.
        """
        data_z0, data_z1, _ = self._split_z(data_z)
        return self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))

    def _continuous_sigma(self, net_output, sigma_key, eps=1e-6):
        soft_key = sigma_key + "_softfloor"
        if soft_key in self.params:
            # Soft variance floor: sigma = sigma_min + softplus-positive
            # learned sd (head stays trainable, bounded below by sigma_min).
            if sigma_key in self.params:
                raise ValueError(
                    f"{sigma_key} and {soft_key} are mutually exclusive"
                )
            sigma_min = tf.cast(float(self.params[soft_key]), tf.float32)
            learned_sd = tf.sqrt(tf.nn.softplus(net_output[:, -1:]) + eps)
            return tf.square(sigma_min + learned_sd)
        if sigma_key in self.params:
            sigma_square = tf.cast(self.params[sigma_key] ** 2, tf.float32)
            sigma_square = tf.ones_like(net_output[:, :1]) * sigma_square
        else:
            sigma_square = tf.nn.softplus(net_output[:, -1:]) + eps
        return sigma_square

    def _gaussian_nll(self, targets, means, sigma_square, event_dim):
        """Compute per-sample Gaussian negative log-likelihood values.

        Parameters
        ----------
        targets : tf.Tensor
            Observations with shape ``(n, event_dim)``.
        means : tf.Tensor
            Conditional means with shape ``(n, event_dim)``.
        sigma_square : tf.Tensor
            Conditional variances with shape ``(n, 1)``.
        event_dim : int
            Dimensionality of the Gaussian observation.

        Returns
        -------
        tf.Tensor
            Pointwise negative log-likelihood values with shape ``(n,)``.
        """
        sq_error = tf.reduce_sum((targets - means) ** 2, axis=1, keepdims=True)
        nll = sq_error / (2.0 * sigma_square) + 0.5 * event_dim * tf.math.log(
            sigma_square
        )
        return tf.squeeze(nll, axis=1)

    def _treatment_mean(self, data_z, data_w):
        treatment_output = self._treatment_output(data_z, data_w)
        mu_x = treatment_output[:, :1]
        if self.params["binary_treatment"]:
            return tf.sigmoid(mu_x)
        return mu_x

    @tf.function
    def _covariate_neg_log_posterior(self, data_v, data_z, eps=1e-6):
        """Average negative log-posterior for the covariate-only latent update."""
        return -tf.reduce_mean(self.get_log_covariate_posterior(data_v, data_z, eps=eps))

    def infer_latent_from_covariates(
        self,
        data_v,
        method=None,
        map_steps=None,
        map_lr=None,
        initial_z=None,
        verbose=0,
    ):
        """Infer latent variables using the covariate-only posterior ``p(z | v)``.

        This is the latent quantity needed by the structural benchmark
        ``f_struct(x, v)`` where the final prediction must not depend on the
        instrument.

        Parameters
        ----------
        data_v : np.ndarray
            Covariates with shape ``(n, v_dim)``.
        method : {"encoder", "map"} or None, optional
            Latent inference rule. ``None`` uses
            ``params["structural_latent_method"]``.
        map_steps : int or None, optional
            Number of gradient steps for the MAP refinement when
            ``method="map"``.
        map_lr : float or None, optional
            Learning rate for the MAP refinement.
        initial_z : np.ndarray or None, optional
            Optional initialization with shape ``(n, sum(z_dims))``.
            By default the encoder output ``e(v)`` is used.
        verbose : int, default=0
            Verbosity level for MAP progress logging.

        Returns
        -------
        np.ndarray
            Latent point estimate with shape ``(n, sum(z_dims))``.
        """
        data_v = self._to_2d_float32(data_v)
        if method is None:
            method = self.params["structural_latent_method"]

        if initial_z is None:
            initial_z = self.e_net(data_v, training=False).numpy()
        else:
            initial_z = self._to_2d_float32(initial_z)

        if method == "encoder":
            return initial_z.astype(np.float32)

        if method != "map":
            raise ValueError("`method` must be one of {'encoder', 'map'}.")

        if map_steps is None:
            map_steps = int(self.params["structural_map_steps"])
        if map_lr is None:
            map_lr = float(self.params["structural_map_lr"])

        if map_steps <= 0:
            return initial_z.astype(np.float32)

        data_v_tf = tf.convert_to_tensor(data_v, dtype=tf.float32)
        data_z = tf.Variable(initial_z.astype(np.float32), trainable=True)
        optimizer = tf.keras.optimizers.Adam(map_lr, beta_1=0.9, beta_2=0.99)

        for step in range(map_steps):
            with tf.GradientTape() as tape:
                loss = self._covariate_neg_log_posterior(data_v_tf, data_z)
            gradients = tape.gradient(loss, [data_z])
            optimizer.apply_gradients(zip(gradients, [data_z]))
            if verbose and ((step + 1) % max(1, map_steps // 5) == 0 or step == 0):
                print(
                    "Structural latent MAP step [%d/%d]: loss_vz = %.4f"
                    % (step + 1, map_steps, float(loss.numpy()))
                )

        return data_z.numpy().astype(np.float32)

    def _sample_treatment(self, data_z, data_w, n_samples=1, use_mean=False, eps=1e-6):
        """Sample treatments from ``p(x | w, z0, z2)``.

        Parameters
        ----------
        data_z : tf.Tensor
            Latent variables with shape ``(n, sum(z_dims))``.
        data_w : tf.Tensor
            Instruments with shape ``(n, w_dim)``.
        n_samples : int, default=1
            Number of Monte Carlo samples per subject.
        use_mean : bool, default=False
            If ``True``, return deterministic mean treatment values instead of
            stochastic samples.

        Returns
        -------
        tuple
            ``(samples, mean_or_prob, variance)`` where ``samples`` has shape
            ``(n_samples, n, 1)``. For binary treatment the third entry is
            ``None``.
        """
        treatment_output = self._treatment_output(data_z, data_w)
        mu_x = treatment_output[:, :1]
        if self.params["binary_treatment"]:
            prob_x = tf.sigmoid(mu_x)
            if use_mean:
                samples = tf.repeat(prob_x[None, :, :], repeats=n_samples, axis=0)
            else:
                uniforms = tf.random.uniform(
                    [n_samples, tf.shape(prob_x)[0], 1], dtype=prob_x.dtype
                )
                samples = tf.cast(uniforms < prob_x[None, :, :], prob_x.dtype)
            return samples, prob_x, None

        sigma_square_x = self._continuous_sigma(
            treatment_output, sigma_key="sigma_x", eps=eps
        )
        if use_mean:
            samples = tf.repeat(mu_x[None, :, :], repeats=n_samples, axis=0)
        else:
            eps_x = tf.random.normal(
                [n_samples, tf.shape(mu_x)[0], 1], dtype=mu_x.dtype
            )
            samples = mu_x[None, :, :] + eps_x * tf.sqrt(sigma_square_x)[None, :, :]
        return samples, mu_x, sigma_square_x

    def _outcome_call_multiplier(self, n_samples):
        if self.params["binary_treatment"]:
            return 2.0
        return float(max(1, n_samples))

    def _outcome_outputs_for_samples(self, data_z, x_samples):
        """Evaluate ``f(z0, z1, x)`` for a stack of treatment samples.

        Parameters
        ----------
        data_z : tf.Tensor
            Latent variables with shape ``(n, sum(z_dims))``.
        x_samples : tf.Tensor
            Treatment samples with shape ``(n_samples, n, 1)``.

        Returns
        -------
        tf.Tensor
            Outcome-network outputs with shape ``(n_samples, n, 2)``.
        """
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
        flat_outputs = self.f_net(flat_inputs)
        return tf.reshape(flat_outputs, (n_samples, n_obs, 2))

    def _integrated_outcome_log_prob(self, data_z, data_w, data_y, n_samples=None, eps=1e-6):
        """Monte Carlo IV log-likelihood ``log p(y | w, z)``.

        Parameters
        ----------
        data_z : tf.Tensor
            Latent variables with shape ``(n, sum(z_dims))``.
        data_w : tf.Tensor
            Instruments with shape ``(n, w_dim)``.
        data_y : tf.Tensor
            Outcomes with shape ``(n, 1)``.
        n_samples : int or None, optional
            Number of treatment Monte Carlo samples used to approximate the
            law-of-total-probability integral for continuous treatment.

        Returns
        -------
        tf.Tensor
            Pointwise integrated log-likelihood values with shape ``(n,)``.
        """
        if self.params["binary_treatment"]:
            treatment_output = self._treatment_output(data_z, data_w)
            prob_x = tf.sigmoid(treatment_output[:, :1])
            x_one = tf.ones_like(data_y)
            x_zero = tf.zeros_like(data_y)

            y_one_output = self._outcome_output(data_z, x_one)
            y_zero_output = self._outcome_output(data_z, x_zero)

            mu_y_one = y_one_output[:, :1]
            mu_y_zero = y_zero_output[:, :1]
            sigma_square_y_one = self._continuous_sigma(
                y_one_output, sigma_key="sigma_y", eps=eps
            )
            sigma_square_y_zero = self._continuous_sigma(
                y_zero_output, sigma_key="sigma_y", eps=eps
            )

            log_prob_one = -(
                (data_y - mu_y_one) ** 2 / (2.0 * sigma_square_y_one)
                + 0.5 * tf.math.log(sigma_square_y_one)
            )
            log_prob_zero = -(
                (data_y - mu_y_zero) ** 2 / (2.0 * sigma_square_y_zero)
                + 0.5 * tf.math.log(sigma_square_y_zero)
            )

            component_log_prob = tf.stack(
                [
                    tf.squeeze(tf.math.log(prob_x + eps) + log_prob_one, axis=1),
                    tf.squeeze(
                        tf.math.log(1.0 - prob_x + eps) + log_prob_zero, axis=1
                    ),
                ],
                axis=0,
            )
            return tf.reduce_logsumexp(component_log_prob, axis=0)

        if n_samples is None:
            n_samples = int(self.params["iv_mc_samples"])
        x_samples, _, _ = self._sample_treatment(
            data_z, data_w, n_samples=n_samples, use_mean=False, eps=eps
        )
        outcome_outputs = self._outcome_outputs_for_samples(data_z, x_samples)
        mu_y = outcome_outputs[:, :, :1]
        # Same noise model as prediction and calibration (fixed override or
        # soft-floored head), so training and inference share one sigma_y.
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

    def _integrated_outcome_mean(self, data_z, data_w, n_samples=None, sample_y=False, eps=1e-6):
        """Monte Carlo expectation of the IV-integrated outcome model.

        Returns the posterior predictive mean of ``Y`` conditional on ``(W, Z)``
        by integrating over the treatment model.
        """
        if self.params["binary_treatment"]:
            treatment_output = self._treatment_output(data_z, data_w)
            prob_x = tf.sigmoid(treatment_output[:, :1])
            x_one = tf.ones([tf.shape(data_z)[0], 1], dtype=tf.float32)
            x_zero = tf.zeros([tf.shape(data_z)[0], 1], dtype=tf.float32)

            y_one_output = self._outcome_output(data_z, x_one)
            y_zero_output = self._outcome_output(data_z, x_zero)
            mu_y_one = y_one_output[:, :1]
            mu_y_zero = y_zero_output[:, :1]

            if not sample_y:
                return prob_x * mu_y_one + (1.0 - prob_x) * mu_y_zero

            sigma_square_y_one = self._continuous_sigma(
                y_one_output, sigma_key="sigma_y", eps=eps
            )
            sigma_square_y_zero = self._continuous_sigma(
                y_zero_output, sigma_key="sigma_y", eps=eps
            )
            take_one = tf.random.uniform(tf.shape(prob_x), dtype=prob_x.dtype) < prob_x
            chosen_mu = tf.where(take_one, mu_y_one, mu_y_zero)
            chosen_sigma = tf.where(take_one, sigma_square_y_one, sigma_square_y_zero)
            return tf.random.normal(
                tf.shape(chosen_mu),
                mean=chosen_mu,
                stddev=tf.sqrt(chosen_sigma),
            )

        if n_samples is None:
            n_samples = int(self.params["eval_mc_samples"])
        x_samples, _, _ = self._sample_treatment(
            data_z, data_w, n_samples=n_samples, use_mean=False, eps=eps
        )
        outcome_outputs = self._outcome_outputs_for_samples(data_z, x_samples)
        mu_y = outcome_outputs[:, :, :1]
        if not sample_y:
            return tf.reduce_mean(mu_y, axis=0)
        sigma_square_y = self._continuous_sigma(
            tf.reshape(outcome_outputs, (-1, 2)),
            sigma_key="sigma_y",
            eps=eps,
        )
        sigma_square_y = tf.reshape(sigma_square_y, tf.shape(mu_y))
        y_components = tf.random.normal(
            tf.shape(mu_y), mean=mu_y, stddev=tf.sqrt(sigma_square_y)
        )
        return tf.reduce_mean(y_components, axis=0)

    @tf.function
    def update_h_net(self, data_z, data_w, data_x, eps=1e-6):
        """Update the treatment generative model ``p(x | w, z0, z2)``.

        Parameters
        ----------
        data_z : tf.Tensor
            Latent variables with shape ``(batch, sum(z_dims))``.
        data_w : tf.Tensor
            Instruments with shape ``(batch, w_dim)``.
        data_x : tf.Tensor
            Treatments with shape ``(batch, 1)``.

        Returns
        -------
        tuple[tf.Tensor, tf.Tensor]
            Negative log-likelihood loss and mean-squared reconstruction error.
        """
        with tf.GradientTape() as gen_tape:
            treatment_output = self._treatment_output(data_z, data_w)
            mu_x = treatment_output[:, :1]

            if self.params["binary_treatment"]:
                loss_x = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, logits=mu_x)
                )
                loss_mse = tf.reduce_mean((data_x - tf.sigmoid(mu_x)) ** 2)
            else:
                sigma_square_x = self._continuous_sigma(
                    treatment_output, sigma_key="sigma_x", eps=eps
                )
                loss_mse = tf.reduce_mean((data_x - mu_x) ** 2)
                loss_x = self._gaussian_nll(
                    data_x, mu_x, sigma_square_x, event_dim=1
                )
                loss_x = tf.reduce_mean(loss_x)

            if self.params["use_bnn"]:
                loss_x += sum(self.h_net.losses) * self.params["kl_weight"]

        h_gradients = gen_tape.gradient(loss_x, self.h_net.trainable_variables)
        self.h_optimizer.apply_gradients(zip(h_gradients, self.h_net.trainable_variables))
        return loss_x, loss_mse

    @tf.function
    def update_f_net(self, data_z, data_w, data_y, n_samples=None, eps=1e-6):
        """Update the outcome model using the IV pseudo-likelihood.

        Parameters
        ----------
        data_z : tf.Tensor
            Latent variables with shape ``(batch, sum(z_dims))``.
        data_w : tf.Tensor
            Instruments with shape ``(batch, w_dim)``.
        data_y : tf.Tensor
            Outcomes with shape ``(batch, 1)``.
        n_samples : int or None, optional
            Number of Monte Carlo treatment samples for the integrated
            likelihood.

        Returns
        -------
        tuple[tf.Tensor, tf.Tensor]
            IV pseudo-likelihood loss and integrated-prediction MSE.
        """
        if n_samples is None:
            n_samples = int(self.params["iv_mc_samples"])

        with tf.GradientTape() as gen_tape:
            log_prob = self._integrated_outcome_log_prob(
                data_z, data_w, data_y, n_samples=n_samples, eps=eps
            )
            loss_y = -tf.reduce_mean(log_prob)
            mean_y = self._integrated_outcome_mean(
                data_z, data_w, n_samples=n_samples, sample_y=False, eps=eps
            )
            loss_mse = tf.reduce_mean((data_y - mean_y) ** 2)

            if self.params["use_bnn"]:
                kl_scale = self._outcome_call_multiplier(n_samples)
                loss_y += (sum(self.f_net.losses) / kl_scale) * self.params["kl_weight"]

        f_gradients = gen_tape.gradient(loss_y, self.f_net.trainable_variables)
        self.f_optimizer.apply_gradients(zip(f_gradients, self.f_net.trainable_variables))
        return loss_y, loss_mse

    @tf.function
    def update_latent_variable_sgd(
        self, data_x, data_y, data_v, data_w, batch_idx, include_outcome=True, eps=1e-6
    ):
        """Update subject-specific latent variables by quasi-posterior ascent.

        The optimized objective is based on
        ``p(z) p(v|z) p(x|w,z) p_iv(y|w,z)`` where the outcome term uses the
        IV integrated pseudo-likelihood.
        """
        with tf.GradientTape() as tape:
            data_z = tf.gather(self.data_z, batch_idx, axis=0)

            g_output = self.g_net(data_z)
            mu_v = g_output[:, : self.params["v_dim"]]
            sigma_square_v = self._continuous_sigma(g_output, sigma_key="sigma_v", eps=eps)
            loss_pv_z = self._gaussian_nll(
                data_v,
                mu_v,
                sigma_square_v,
                event_dim=self.params["v_dim"],
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
                    treatment_output, sigma_key="sigma_x", eps=eps
                )
                loss_px_z = self._gaussian_nll(
                    data_x, mu_x, sigma_square_x, event_dim=1
                )
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
    def train_gen_step(self, data_z, data_v, data_w, data_x, data_y):
        """One EGM warm-start step for ``(g, e, h, f)``.

        The warm-start uses deconfounded treatment predictions from the
        treatment model before fitting the outcome network.
        """
        with tf.GradientTape(persistent=True) as gen_tape:
            sigma_square_loss = 0.0
            g_output = self.g_net(data_z)
            data_v_ = g_output[:, : self.params["v_dim"]]
            sigma_square_loss += tf.reduce_mean(tf.square(g_output[:, -1]))

            data_z_ = self.e_net(data_v)
            data_z0, data_z1, data_z2 = self._split_z(data_z_)

            data_z__ = self.e_net(data_v_)
            data_v__ = self.g_net(data_z_)[:, : self.params["v_dim"]]
            data_dz_ = self.dz_net(data_z_)

            l2_loss_v = tf.reduce_mean((data_v - data_v__) ** 2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__) ** 2)
            e_loss_adv = -tf.reduce_mean(data_dz_)

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
    def train_gen_step_integral(self, data_z, data_v, data_w, data_x, data_y):
        """EGM warm-start step with the integrated outcome moment condition.

        Replaces the naive plug-in ``f(z, mu_x)`` by the first-moment
        condition ``(y - E[f(z, x) | w, e(v)])**2`` of the IV
        pseudo-likelihood. The 1-D Gaussian integral is evaluated with
        Gauss-Hermite quadrature (deterministic; exact for polynomials of
        degree <= 2*nodes - 1). Binary treatment uses the exact two-point
        integral instead of inserting a fractional probability into f.
        """
        with tf.GradientTape(persistent=True) as gen_tape:
            sigma_square_loss = 0.0
            g_output = self.g_net(data_z)
            data_v_ = g_output[:, : self.params["v_dim"]]
            sigma_square_loss += tf.reduce_mean(tf.square(g_output[:, -1]))

            data_z_ = self.e_net(data_v)
            data_z0, data_z1, data_z2 = self._split_z(data_z_)

            data_z__ = self.e_net(data_v_)
            data_v__ = self.g_net(data_z_)[:, : self.params["v_dim"]]
            data_dz_ = self.dz_net(data_z_)

            l2_loss_v = tf.reduce_mean((data_v - data_v__) ** 2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__) ** 2)
            e_loss_adv = -tf.reduce_mean(data_dz_)

            h_output = self.h_net(tf.concat([data_z0, data_z2, data_w], axis=-1))
            data_x_ = h_output[:, :1]
            if self.params["binary_treatment"]:
                l2_loss_x = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=data_x, logits=data_x_
                    )
                )
                prob_x = tf.sigmoid(data_x_)
                if str(self.params["egm_outcome_grad_path"]) == "stop":
                    prob_x = tf.stop_gradient(prob_x)
                f_out_one = self.f_net(
                    tf.concat([data_z0, data_z1, tf.ones_like(data_x_)], axis=-1)
                )
                f_out_zero = self.f_net(
                    tf.concat([data_z0, data_z1, tf.zeros_like(data_x_)], axis=-1)
                )
                data_y_ = prob_x * f_out_one[:, :1] + (1.0 - prob_x) * f_out_zero[:, :1]
                sigma_square_loss += tf.reduce_mean(tf.square(f_out_one[:, -1]))
                sigma_square_loss += tf.reduce_mean(tf.square(f_out_zero[:, -1]))
            else:
                sigma_square_loss += tf.reduce_mean(tf.square(h_output[:, -1]))
                l2_loss_x = tf.reduce_mean((data_x_ - data_x) ** 2)

                sigma_mode = self.params["egm_outcome_sigma"]
                if sigma_mode == "head":
                    sigma_square_x = self._continuous_sigma(
                        h_output, sigma_key="sigma_x"
                    )
                elif sigma_mode == "residual_ema":
                    batch_resid = tf.stop_gradient(
                        tf.reduce_mean(tf.square(data_x - data_x_))
                    )
                    self.egm_sigma2_x_ema.assign(
                        0.99 * self.egm_sigma2_x_ema + 0.01 * batch_resid
                    )
                    sigma_square_x = self.egm_sigma2_x_ema * tf.ones_like(data_x_)
                else:
                    sigma_square_x = float(sigma_mode) ** 2 * tf.ones_like(data_x_)
                cap = float(self.params["egm_outcome_sigma_cap"])
                sigma_square_x = tf.minimum(sigma_square_x, cap ** 2)

                # mu keeps its gradient path into h (same as the plug-in), unless egm_outcome_grad_path='stop'
                # sigma is stop-gradient so f cannot lower the loss by
                # inflating the integration width.
                sigma_x = tf.stop_gradient(tf.sqrt(sigma_square_x))
                if str(self.params["egm_outcome_grad_path"]) == "stop":
                    x_center = tf.stop_gradient(data_x_)
                else:
                    x_center = data_x_
                x_nodes = (
                    x_center[None, :, :]
                    + 1.4142135623730951 * sigma_x[None, :, :] * self._egm_gh_t
                )
                f_outputs = self._outcome_outputs_for_samples(data_z_, x_nodes)
                data_y_ = tf.reduce_sum(self._egm_gh_w * f_outputs[:, :, :1], axis=0)
                sigma_square_loss += tf.reduce_mean(tf.square(f_outputs[:, :, -1]))

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

    def egm_init(
        self,
        data,
        egm_n_iter=30000,
        batch_size=32,
        egm_batches_per_eval=500,
        verbose=1,
        evaluation_callback=None,
    ):
        """Run IV-aware EGM initialization on ``(x, y, v, w)`` data."""
        data_x, data_y, data_v, data_w = self._parse_train_data(data)

        print("EGM Initialization Starts ...")
        use_integral = str(self.params.get("egm_outcome_loss", "plugin")) == "integral"
        overrides_plugin = type(self).train_gen_step is not BGM_IV.train_gen_step
        has_own_integral = (
            type(self).train_gen_step_integral is not BGM_IV.train_gen_step_integral
        )
        if use_integral and overrides_plugin and not has_own_integral:
            raise NotImplementedError(
                "egm_outcome_loss='integral' pairs with the base BGM_IV EGM step; "
                f"{type(self).__name__} overrides train_gen_step and needs its own "
                "integral variant before this flag can be enabled."
            )
        egm_gen_step = self.train_gen_step_integral if use_integral else self.train_gen_step
        for batch_iter in range(egm_n_iter + 1):
            for _ in range(self.params["g_d_freq"]):
                batch_idx = np.random.choice(len(data_x), batch_size, replace=False)
                batch_z = self.z_sampler.get_batch(batch_size)
                batch_v = data_v[batch_idx, :]
                dz_loss, d_loss = self.train_disc_step(batch_z, batch_v)

            batch_z = self.z_sampler.get_batch(batch_size)
            batch_idx = np.random.choice(len(data_x), batch_size, replace=False)
            batch_x = data_x[batch_idx, :]
            batch_y = data_y[batch_idx, :]
            batch_v = data_v[batch_idx, :]
            batch_w = data_w[batch_idx, :]
            e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss = (
                egm_gen_step(batch_z, batch_v, batch_w, batch_x, batch_y)
            )
            if batch_iter % egm_batches_per_eval == 0:
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
                causal_pre, _, _, _ = self.evaluate(data=data)
                causal_pre = causal_pre.numpy()
                if self.params["save_res"]:
                    save_data(
                        "{}/causal_pre_egm_init_iter-{}.txt".format(
                            self.save_dir, batch_iter
                        ),
                        causal_pre,
                    )
        _, mse_x, mse_y, mse_v = self.evaluate(data=data, data_z=None)
        record = self._record_training_metrics(
            stage="post_egm",
            epoch=None,
            mse_x=float(mse_x),
            mse_y=float(mse_y),
            mse_v=float(mse_v),
            evaluation_callback=evaluation_callback,
            include_outcome=True,
        )
        if verbose:
            print("Post-EGM encoder metrics:", self._format_training_record(record))
        print("EGM Initialization Ends.")

    def _apply_bn_determinism(self):
        """Force non-fused BatchNormalization under the determinism contract.

        FusedBatchNormGradV3 has no deterministic GPU implementation, so with
        ``deterministic_training`` every BatchNormalization layer must take
        the non-fused path.  Must run BEFORE the layers are built (``fused``
        is resolved at build time); fit() calls this ahead of training.
        """
        if not bool(self.params.get("deterministic_training", False)):
            return
        for attr in vars(self).values():
            if isinstance(attr, tf.Module):
                for layer in getattr(attr, "submodules", ()):
                    if isinstance(layer, tf.keras.layers.BatchNormalization):
                        layer.fused = False

    def _outcome_to_particles_weight(self):
        """Resolve gamma in [0, 1] for the outcome->particle coupling.

        The particle objective is loss_pv_z + loss_prior_z + loss_px_z +
        gamma * loss_py_z.  An explicit `outcome_to_particles_weight` wins;
        otherwise the boolean alias `stop_outcome_to_particles` maps
        True -> 0.0 (severed C-form) and False -> 1.0 (joint objective).
        """
        if "outcome_to_particles_weight" in self.params:
            gamma = float(self.params["outcome_to_particles_weight"])
            if not 0.0 <= gamma <= 1.0:
                raise ValueError(
                    "outcome_to_particles_weight must be in [0, 1]"
                )
            return gamma
        if bool(self.params.get("stop_outcome_to_particles", False)):
            return 0.0
        return 1.0

    def fit(
        self,
        data,
        epochs=100,
        epochs_per_eval=5,
        batch_size=32,
        startoff=0,
        use_egm_init=True,
        egm_n_iter=30000,
        egm_batches_per_eval=500,
        save_format="txt",
        verbose=1,
        first_stage_warmup_epochs=None,
        evaluation_callback=None,
    ):
        """Train ``BGM_IV`` on observed data ``(x, y, v, w)``.

        Parameters
        ----------
        data : tuple of np.ndarray
            Training tuple ``(x, y, v, w)`` with shapes
            ``(n,1)``, ``(n,1)``, ``(n,v_dim)``, and ``(n,w_dim)``.
        epochs, epochs_per_eval, batch_size, startoff, use_egm_init,
        egm_n_iter, egm_batches_per_eval, save_format, verbose :
            Same meanings as in :class:`CausalBGM`.
        first_stage_warmup_epochs : int or None, optional
            Number of epochs to train only ``p(v|z)`` and ``p(x|w,z)`` before
            switching on the IV outcome pseudo-likelihood.
        """
        self._apply_bn_determinism()
        data_x, data_y, data_v, data_w = self._parse_train_data(data)
        if first_stage_warmup_epochs is None:
            first_stage_warmup_epochs = int(self.params["first_stage_warmup_epochs"])

        if self.params["save_res"]:
            with open("{}/params.txt".format(self.save_dir), "w") as f_params:
                f_params.write(str(self.params))

        if use_egm_init:
            self.training_history = []
            self.egm_init(
                data,
                egm_n_iter=egm_n_iter,
                egm_batches_per_eval=egm_batches_per_eval,
                batch_size=batch_size,
                verbose=verbose,
                evaluation_callback=evaluation_callback,
            )
            print("Initialize latent variables Z with e(V)...")
            data_z_init = self.e_net(data_v)
        else:
            self.training_history = []
            print("Random initialization of latent variables Z...")
            data_z_init = np.random.normal(
                0,
                1,
                size=(len(data_x), sum(self.params["z_dims"])),
            ).astype("float32")

        self.data_z = tf.Variable(data_z_init, name="Latent Variable", trainable=True)
        # Register the particles in the checkpoint so trained particles are
        # recoverable (attribute assignment adds the dependency to self.ckpt).
        self.ckpt.data_z = self.data_z

        if self.params.get("save_post_egm_checkpoint", False) and self.params["save_model"]:
            # Ablation arm A: persist the PURE post-EGM state before any BGM
            # update.  The regular per-epoch saves happen at epoch END and are
            # rotated by max_to_keep, so no pure-EGM snapshot survives without
            # this.  Pair with fit_epochs=0 + fit_first_stage_warmup_epochs=1
            # so the loop neither trains with outcome nor overwrites this save.
            post_egm_path = self.ckpt_manager.save(0)
            print(f"Saving post-EGM checkpoint at {post_egm_path}")

        show_batch_progress = bool(self.params.get("fit_use_progress_bar", False)) and bool(verbose)
        print("Iterative Updating Starts ...")
        for epoch in range(epochs + 1):
            sample_idx = np.random.choice(len(data_x), len(data_x), replace=False)
            include_outcome = epoch >= first_stage_warmup_epochs
            # Outcome->particle coupling gamma: the particle objective is
            # loss_pv_z + loss_prior_z + loss_px_z + gamma * loss_py_z.
            # gamma=0 severs ONLY the outcome->particle gradient path (C-form)
            # while f keeps its outcome training (update_f_net stays gated by
            # include_outcome alone); gamma=1 is the joint objective (B-form).
            # EGM is untouched, so a deterministic same-seed run forks from an
            # identical ckpt-0 at every gamma.
            gamma = self._outcome_to_particles_weight()
            particle_outcome = include_outcome and gamma > 0.0

            batch_iterator = range(0, len(data_x), batch_size)
            if show_batch_progress:
                batch_iterator = tqdm(
                    batch_iterator,
                    total=int(np.ceil(len(data_x) / batch_size)),
                    desc=f"Epoch {epoch}/{epochs}",
                    unit="batch",
                )
            for i in batch_iterator:
                    batch_idx = sample_idx[i : i + batch_size]
                    batch_z = tf.gather(self.data_z, batch_idx, axis=0)
                    batch_x = data_x[batch_idx, :]
                    batch_y = data_y[batch_idx, :]
                    batch_v = data_v[batch_idx, :]
                    batch_w = data_w[batch_idx, :]

                    loss_v, loss_mse_v = self.update_g_net(batch_z, batch_v)
                    loss_x, loss_mse_x = self.update_h_net(batch_z, batch_w, batch_x)

                    if include_outcome:
                        loss_y, loss_mse_y = self.update_f_net(
                            batch_z,
                            batch_w,
                            batch_y,
                            n_samples=int(self.params["iv_mc_samples"]),
                        )
                    else:
                        loss_y = tf.constant(0.0, dtype=tf.float32)
                        loss_mse_y = tf.constant(0.0, dtype=tf.float32)

                    loss_posterior_z = self.update_latent_variable_sgd(
                        batch_x,
                        batch_y,
                        batch_v,
                        batch_w,
                        batch_idx,
                        include_outcome=particle_outcome,
                    )

                    loss_contents = (
                        "loss_px_wz: [%.4f], loss_mse_x: [%.4f], loss_py_iv: [%.4f], "
                        "loss_mse_y: [%.4f], loss_pv_z: [%.4f], loss_mse_v: [%.4f], "
                        "loss_posterior_z: [%.4f]"
                        % (
                            loss_x,
                            loss_mse_x,
                            loss_y,
                            loss_mse_y,
                            loss_v,
                            loss_mse_v,
                            loss_posterior_z,
                        )
                    )
                    if show_batch_progress:
                        batch_iterator.set_postfix_str(loss_contents)

            if epoch % epochs_per_eval == 0:
                causal_pre, mse_x, mse_y, mse_v = self.evaluate(
                    data=data, data_z=self.data_z
                )
                causal_pre = causal_pre.numpy()
                record = self._record_training_metrics(
                    stage="epoch_eval",
                    epoch=epoch,
                    mse_x=float(mse_x),
                    mse_y=float(mse_y),
                    mse_v=float(mse_v),
                    evaluation_callback=evaluation_callback,
                    include_outcome=include_outcome,
                )

                if verbose:
                    print(
                        "Epoch [%d/%d]: %s\n"
                        % (epoch, epochs, self._format_training_record(record))
                    )

                if include_outcome and epoch >= startoff:
                    if self.params["save_model"]:
                        ckpt_save_path = self.ckpt_manager.save(epoch)
                        print(
                            "Saving checkpoint for epoch {} at {}".format(
                                epoch, ckpt_save_path
                            )
                        )

                if self.params["save_res"]:
                    save_data(
                        "{}/causal_pre_at_{}.{}".format(
                            self.save_dir, epoch, save_format
                        ),
                        causal_pre,
                    )
        return self.training_history

    @tf.function
    def evaluate(self, data, data_z=None, nb_intervals=200):
        """Evaluate training reconstruction metrics and causal summaries.

        Parameters
        ----------
        data : tuple of np.ndarray
            Data tuple ``(x, y, v, w)``.
        data_z : tf.Tensor or np.ndarray or None, optional
            Optional latent matrix with shape ``(n, sum(z_dims))``. If omitted,
            the encoder output ``e(v)`` is used.
        nb_intervals : int, default=200
            Number of treatment values used for continuous-treatment ADRF
            summaries.

        Returns
        -------
        tuple
            ``(causal_summary, mse_x, mse_y, mse_v)`` where the first item is
            the ITE vector for binary treatment or ADRF curve for continuous
            treatment.
        """
        data_x, data_y, data_v, data_w = data
        if data_z is None:
            data_z = self.e_net(data_v)

        data_z0, data_z1, _ = self._split_z(data_z)
        data_v_pred = self.g_net(data_z)[:, : self.params["v_dim"]]
        data_x_pred = self._treatment_mean(data_z, data_w)
        data_y_pred = self._integrated_outcome_mean(
            data_z,
            data_w,
            n_samples=int(self.params["eval_mc_samples"]),
            sample_y=False,
        )

        mse_v = tf.reduce_mean((data_v - data_v_pred) ** 2)
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

    def predict_structural(
        self,
        data_x,
        data_v,
        data_z=None,
        sample_y=False,
        latent_method=None,
        map_steps=None,
        map_lr=None,
        verbose=0,
    ):
        """Predict the structural outcome using treatment and covariates only.

        This matches the DFIV/DeepIV evaluation target:
        ``f_struct(x, v) = E[Y | do(X=x), V=v]``.

        Parameters
        ----------
        data_x : np.ndarray
            Treatment values with shape ``(n, 1)``.
        data_v : np.ndarray
            Observed covariates with shape ``(n, v_dim)``.
        data_z : np.ndarray or None, optional
            Optional latent point estimate with shape ``(n, sum(z_dims))``.
            If provided, latent inference is skipped.
        sample_y : bool, default=False
            If ``True``, sample from the outcome variance head.
        latent_method : {"encoder", "map"} or None, optional
            Latent inference rule used when ``data_z`` is not supplied.
        map_steps, map_lr :
            MAP-specific hyper-parameters.

        Returns
        -------
        np.ndarray
            Point prediction with shape ``(n, 1)``.
        """
        data_x = self._to_2d_float32(data_x)
        data_v = self._to_2d_float32(data_v)
        resolved_method = latent_method or self.params["structural_latent_method"]
        if data_z is None:
            data_z = self.infer_latent_from_covariates(
                data_v,
                method=resolved_method,
                map_steps=map_steps,
                map_lr=map_lr,
                verbose=verbose,
            )
        data_z = tf.convert_to_tensor(data_z, dtype=tf.float32)
        data_z0, data_z1, _ = self._split_z(data_z)
        outcome_output = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))
        mu_y = outcome_output[:, :1]
        if not sample_y:
            return mu_y.numpy()
        sigma_square_y = self._continuous_sigma(outcome_output, sigma_key="sigma_y")
        y_draw = tf.random.normal(
            tf.shape(mu_y), mean=mu_y, stddev=tf.sqrt(sigma_square_y)
        )
        return y_draw.numpy()

    def evaluate_structural_mse(
        self,
        data_x,
        data_v,
        data_y_true,
        data_z=None,
        latent_method=None,
        map_steps=None,
        map_lr=None,
        verbose=0,
    ):
        """Evaluate structural-function MSE for the DFIV/DeepIV benchmark.

        Parameters
        ----------
        data_x : np.ndarray
            Treatment grid with shape ``(n, 1)``.
        data_v : np.ndarray
            Observed covariates with shape ``(n, v_dim)``.
        data_y_true : np.ndarray
            Ground-truth structural outcomes with shape ``(n, 1)``.
        data_z : np.ndarray or None, optional
            Optional latent point estimate with shape ``(n, sum(z_dims))``.
        latent_method : {"encoder", "map"} or None, optional
            Latent inference rule when ``data_z`` is not provided.
        map_steps, map_lr :
            MAP-specific hyper-parameters.

        Returns
        -------
        float
            Mean squared error of the structural point predictor.
        """
        data_y_true = self._to_2d_float32(data_y_true)
        data_y_pred = self.predict_structural(
            data_x,
            data_v,
            data_z=data_z,
            latent_method=latent_method,
            map_steps=map_steps,
            map_lr=map_lr,
            verbose=verbose,
        )
        return float(np.mean((data_y_true - data_y_pred) ** 2))

    @tf.function
    def get_log_covariate_posterior(self, data_v, data_z, eps=1e-6):
        """Log covariate-only posterior ``log p(z | v)`` up to a constant.

        Parameters
        ----------
        data_v : tf.Tensor
            Covariates with shape ``(n, v_dim)``.
        data_z : tf.Tensor
            Latent variables with shape ``(n, sum(z_dims))``.

        Returns
        -------
        tf.Tensor
            Pointwise log posterior values proportional to
            ``log p(z) + log p(v | z)`` with shape ``(n,)``.
        """
        g_output = self.g_net(data_z)
        mu_v = g_output[:, : self.params["v_dim"]]
        sigma_square_v = self._continuous_sigma(g_output, sigma_key="sigma_v", eps=eps)
        loss_pv_z = self._gaussian_nll(
            data_v,
            mu_v,
            sigma_square_v,
            event_dim=self.params["v_dim"],
        )
        loss_prior_z = tf.reduce_sum(data_z ** 2, axis=1) / 2.0
        return -(loss_pv_z + loss_prior_z)
