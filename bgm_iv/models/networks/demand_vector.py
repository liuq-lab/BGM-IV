import tensorflow as tf


class DemandVectorFeatureExtractor(tf.keras.Model):
    """Extract features from `(time, high-dimensional vector proxy)`."""

    def __init__(
        self,
        v_dim=785,
        vector_dim=784,
        num_dense_features=1,
        vector_feature_dim=64,
        name="demand_vector_feature_extractor",
    ):
        super().__init__(name=name)
        self.v_dim = int(v_dim)
        self.vector_dim = int(vector_dim)
        self.num_dense_features = int(num_dense_features)
        if self.v_dim != self.num_dense_features + self.vector_dim:
            raise ValueError("DemandVectorFeatureExtractor requires `v_dim == num_dense_features + vector_dim`.")

        self.vector_hidden = tf.keras.layers.Dense(128)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.vector_output = tf.keras.layers.Dense(int(vector_feature_dim))

    def call(self, data, training=True):
        dense_feature = tf.cast(data[:, : self.num_dense_features], tf.float32)
        vector_start = self.num_dense_features
        vector_end = vector_start + self.vector_dim
        vector_feature = tf.cast(data[:, vector_start:vector_end], tf.float32)
        vector_feature = tf.nn.relu(self.vector_hidden(vector_feature))
        vector_feature = self.dropout(vector_feature, training=training)
        vector_feature = self.vector_output(vector_feature)
        return tf.concat([dense_feature, vector_feature], axis=1)


class DemandVectorEncoder(tf.keras.Model):
    """Encode `(time, vector proxy)` covariates into latent variables."""

    def __init__(
        self,
        z_dim,
        v_dim=785,
        vector_dim=784,
        nb_units=(128, 64),
        vector_feature_dim=64,
        name="demand_vector_encoder",
    ):
        super().__init__(name=name)
        self.v_dim = int(v_dim)
        self.vector_dim = int(vector_dim)
        self.feature_extractor = DemandVectorFeatureExtractor(
            v_dim=self.v_dim,
            vector_dim=self.vector_dim,
            num_dense_features=1,
            vector_feature_dim=vector_feature_dim,
            name=f"{name}_feature_extractor",
        )
        self.hidden_layers = [tf.keras.layers.Dense(int(units)) for units in nb_units]
        self.output_layer = tf.keras.layers.Dense(int(z_dim))

    def call(self, data, training=True):
        x = self.feature_extractor(data, training=training)
        for layer in self.hidden_layers:
            x = tf.nn.leaky_relu(layer(x), alpha=0.2)
        return self.output_layer(x)


class DemandVectorCovariateDecoder(tf.keras.Model):
    """Decode latent variables into `(time, vector proxy)` covariates."""

    def __init__(
        self,
        z_dim,
        v_dim=785,
        vector_dim=784,
        name="demand_vector_covariate_decoder",
    ):
        super().__init__(name=name)
        self.v_dim = int(v_dim)
        self.vector_dim = int(vector_dim)
        if self.v_dim != 1 + self.vector_dim:
            raise ValueError("DemandVectorCovariateDecoder requires `v_dim == 1 + vector_dim`.")

        self.time_hidden = tf.keras.layers.Dense(64)
        self.time_mean_head = tf.keras.layers.Dense(1)
        self.time_var_head = tf.keras.layers.Dense(1)

        self.vector_hidden_1 = tf.keras.layers.Dense(128)
        self.vector_hidden_2 = tf.keras.layers.Dense(64)
        self.vector_mean_head = tf.keras.layers.Dense(self.vector_dim)
        self.vector_var_head = tf.keras.layers.Dense(self.vector_dim)

    def call(self, data_z, training=True):
        del training
        time_hidden = tf.nn.leaky_relu(self.time_hidden(data_z), alpha=0.2)
        time_mean = self.time_mean_head(time_hidden)
        time_var = tf.nn.softplus(self.time_var_head(time_hidden)) + 1e-6

        vector_hidden = tf.nn.leaky_relu(self.vector_hidden_1(data_z), alpha=0.2)
        vector_hidden = tf.nn.leaky_relu(self.vector_hidden_2(vector_hidden), alpha=0.2)
        vector_mean = self.vector_mean_head(vector_hidden)
        vector_var = tf.nn.softplus(self.vector_var_head(vector_hidden)) + 1e-6
        public_v = tf.concat([time_mean, vector_mean], axis=1)

        return {
            "time_mean": time_mean,
            "time_var": time_var,
            "vector_mean": vector_mean,
            "vector_var": vector_var,
            "public_v": public_v,
        }
