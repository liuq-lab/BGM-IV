import tensorflow as tf


class DemandImageFeatureExtractor(tf.keras.Model):
    """Extract features from `(time, flattened MNIST image, optional noise)`."""

    def __init__(
        self,
        v_dim=785,
        num_dense_features=1,
        image_feature_dim=64,
        noise_feature_dim=32,
        filters=64,
        name="demand_image_feature_extractor",
    ):
        super().__init__(name=name)
        self.v_dim = int(v_dim)
        if self.v_dim < 785:
            raise ValueError("DemandImageFeatureExtractor requires `v_dim >= 785`.")
        self.num_dense_features = int(num_dense_features)
        self.image_feature_dim = int(image_feature_dim)
        self.image_dim = 28 * 28
        self.extra_noise_dim = self.v_dim - self.num_dense_features - self.image_dim
        if self.extra_noise_dim < 0:
            raise ValueError("`v_dim` is too small for `(time, image)` covariates.")

        self.conv1 = tf.keras.layers.Conv2D(filters, 3, padding="valid", use_bias=False)
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding="valid", use_bias=False)
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=2)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.flatten = tf.keras.layers.Flatten()
        self.linear1 = tf.keras.layers.Dense(128)
        self.linear2 = tf.keras.layers.Dense(self.image_feature_dim)
        self.noise_hidden = None
        self.noise_output = None
        if self.extra_noise_dim > 0:
            self.noise_hidden = tf.keras.layers.Dense(64)
            self.noise_output = tf.keras.layers.Dense(int(noise_feature_dim))

    @staticmethod
    def normalize_image_pixels(image_flat):
        return tf.cast(image_flat, tf.float32) / 255.0

    def call(self, data, training=True):
        dense_feature = tf.cast(data[:, : self.num_dense_features], tf.float32)
        image_start = self.num_dense_features
        image_end = image_start + self.image_dim
        image_flat = self.normalize_image_pixels(data[:, image_start:image_end])
        image = tf.reshape(image_flat, (-1, 28, 28, 1))

        image_feature = tf.nn.relu(self.conv1(image))
        image_feature = self.maxpool(tf.nn.relu(self.conv2(image_feature)))
        image_feature = self.flatten(image_feature)
        image_feature = self.dropout1(image_feature, training=training)
        image_feature = self.dropout2(
            tf.nn.relu(self.linear1(image_feature)),
            training=training,
        )
        image_feature = self.linear2(image_feature)
        features = [dense_feature, image_feature]
        if self.extra_noise_dim > 0:
            noise_feature = tf.cast(data[:, image_end:], tf.float32)
            noise_feature = tf.nn.leaky_relu(self.noise_hidden(noise_feature), alpha=0.2)
            noise_feature = self.noise_output(noise_feature)
            features.append(noise_feature)
        return tf.concat(features, axis=1)


class DemandImageEncoder(tf.keras.Model):
    """Encode `(time, image)` covariates into latent variables."""

    def __init__(
        self,
        z_dim,
        v_dim=785,
        nb_units=(128, 64),
        image_feature_dim=64,
        noise_feature_dim=32,
        name="demand_image_encoder",
    ):
        super().__init__(name=name)
        self.v_dim = int(v_dim)
        self.feature_extractor = DemandImageFeatureExtractor(
            v_dim=self.v_dim,
            num_dense_features=1,
            image_feature_dim=image_feature_dim,
            noise_feature_dim=noise_feature_dim,
            name=f"{name}_feature_extractor",
        )
        self.hidden_layers = [tf.keras.layers.Dense(int(units)) for units in nb_units]
        self.output_layer = tf.keras.layers.Dense(int(z_dim))

    def call(self, data, training=True):
        x = self.feature_extractor(data, training=training)
        for layer in self.hidden_layers:
            x = tf.nn.leaky_relu(layer(x), alpha=0.2)
        return self.output_layer(x)


class DemandImageCovariateDecoder(tf.keras.Model):
    """Decode latent variables into `(time, image)` covariates."""

    def __init__(
        self,
        z_dim,
        v_dim=785,
        filters=32,
        name="demand_image_covariate_decoder",
    ):
        super().__init__(name=name)
        self.v_dim = int(v_dim)
        self.image_dim = 28 * 28
        if self.v_dim < 1 + self.image_dim:
            raise ValueError("DemandImageCovariateDecoder requires `v_dim >= 785`.")
        self.extra_noise_dim = self.v_dim - 1 - self.image_dim
        self.time_hidden = tf.keras.layers.Dense(64)
        self.time_mean_head = tf.keras.layers.Dense(1)
        self.time_var_head = tf.keras.layers.Dense(1)

        self.image_fc = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(int(z_dim),)),
                tf.keras.layers.Dense(7 * 7 * filters * 4),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Reshape((7, 7, filters * 4)),
            ]
        )
        self.image_up = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose(filters * 2, 3, strides=2, padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(0.2),
            ]
        )
        self.image_logits_head = tf.keras.layers.Conv2D(1, 1, padding="same", name="image_logits")
        self.noise_hidden = None
        self.noise_mean_head = None
        self.noise_var_head = None
        if self.extra_noise_dim > 0:
            self.noise_hidden = tf.keras.layers.Dense(64)
            self.noise_mean_head = tf.keras.layers.Dense(self.extra_noise_dim)
            self.noise_var_head = tf.keras.layers.Dense(self.extra_noise_dim)

    def call(self, data_z, training=True):
        time_hidden = tf.nn.leaky_relu(self.time_hidden(data_z), alpha=0.2)
        time_mean = self.time_mean_head(time_hidden)
        time_var = tf.nn.softplus(self.time_var_head(time_hidden)) + 1e-6

        image_hidden = self.image_fc(data_z, training=training)
        image_hidden = self.image_up(image_hidden, training=training)
        image_logits = self.image_logits_head(image_hidden)
        image_probs = tf.nn.sigmoid(image_logits)
        image_logits_flat = tf.reshape(image_logits, (-1, self.image_dim))
        image_probs_flat = tf.reshape(image_probs, (-1, self.image_dim))
        if self.extra_noise_dim > 0:
            noise_hidden = tf.nn.leaky_relu(self.noise_hidden(data_z), alpha=0.2)
            noise_mean = self.noise_mean_head(noise_hidden)
            noise_var = tf.nn.softplus(self.noise_var_head(noise_hidden)) + 1e-6
        else:
            batch_size = tf.shape(data_z)[0]
            noise_mean = tf.zeros((batch_size, 0), dtype=tf.float32)
            noise_var = tf.zeros((batch_size, 0), dtype=tf.float32)
        public_v = tf.concat([time_mean, image_probs_flat * 255.0, noise_mean], axis=1)

        return {
            "time_mean": time_mean,
            "time_var": time_var,
            "image_logits": image_logits_flat,
            "image_probs": image_probs_flat,
            "noise_mean": noise_mean,
            "noise_var": noise_var,
            "public_v": public_v,
        }
