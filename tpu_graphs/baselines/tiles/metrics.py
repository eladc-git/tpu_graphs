# Copyright 2023 The tpu_graphs Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for computing eval metrics and loss functions."""

from typing import Any, Callable, Sequence
import tensorflow as tf
import tensorflow_ranking as tfr


def top_error_performance(
        test_ds: tf.data.Dataset,
        model_fn: Callable[[Any, int], tf.Tensor],
        top_k: Sequence[int] = (1, 5, 10)):
    """Computes test errors as: (ModelChosen - Best) / Best.

  where ModelChosen is the (measured) runtime of configuration chosen by model
  (we select the least of the top-k of the model) and Best is the minimum
  runtime of all configurations.

  Args:
    test_ds: Dataset which the metrics are computed over. It should yield
      GraphTensor instances that will be passed to model_fn (on its first
      argument).
    model_fn: Function that accept GraphTensor instances (yield from test_ds)
      and integer of `N` "number of module configurations". The function should
      return a tensor with shape `[1, N]`, with each of the `N` scores are
      model's estimates of runtime: one scalar per configuration. Only order is
      important but not the actual value.
    top_k: For each k in this list, we select the ground-truth runtimes
      corresponding to least `k` values, and use as ModelChosen to compute above
      formula.

  Returns:
    dict with each of `top_k` as a key and value is error described above.
  """
    num_examples = 0
    result: dict[int, float] = {k: 0.0 for k in top_k}
    tile_metric = 0
    for graph in test_ds:
        num_configs = int(graph.node_sets['config'].sizes[0].numpy())
        preds = model_fn(graph, num_configs)
        preds = tf.squeeze(preds, 0)  # Remove batch size (of 1)
        sorted_indices = tf.argsort(preds)
        runtimes = graph.node_sets['config']['runtimes']
        time_best = tf.reduce_min(runtimes)
        num_examples += 1
        for k in top_k:
            time_model_candidates = tf.gather(runtimes, sorted_indices[:k])
            best_of_candidates = tf.reduce_min(time_model_candidates)
            result[k] += float((best_of_candidates - time_best) / time_best)
            if k==5:
                tile_metric_i = 2 - float(best_of_candidates / time_best)
                if tile_metric_i < 0.8:
                    id = graph.node_sets['g']['tile_id'][0]
                    print(f'{id}:{tile_metric_i}')
                tile_metric += tile_metric_i


    for k in top_k:
        result[k] /= num_examples

    tile_metric /= num_examples

    return result, tile_metric


LOSS_DICT = {
    'ListMLELoss': tfr.keras.losses.ListMLELoss(temperature=10),
    'PairwiseHingeLoss': tfr.keras.losses.PairwiseHingeLoss(temperature=10),
    'MSE': tf.keras.losses.MeanSquaredError(),
}


def parse_loss_str(loss_str: str) -> list[tuple[float, tf.keras.losses.Loss]]:
    """Parses string-encoded loss names and weights into objects and floats.

  Args:
    loss_str (str): comma-separated string with items "lossName:lossWeight"
      where lossName must be in dict `LOSS_DICT` and lossWeight must contain
      float.

  Returns:
    List of loss objects and their weights.
  """
    weighted_losses: list[tuple[float, tf.keras.losses.Loss]] = []
    for loss_str in loss_str.split(','):
        loss_name, loss_weight = loss_str.split(':')
        assert loss_name in LOSS_DICT
        loss_weight = float(loss_weight)
        loss = LOSS_DICT[loss_name]
        weighted_losses.append((loss_weight, loss))
    return weighted_losses


class CombinedLoss(tf.keras.losses.Loss):
    """Computes loss as a weighted-average of losses."""

    def __init__(
            self,
            weighted_losses: 'None|list[tuple[float, tf.keras.losses.Loss]]' = None,
            reduction=None, name=None):
        super().__init__()
        if weighted_losses is None:
            weighted_losses = [(1.0, tfr.keras.losses.ListMLELoss(temperature=10))]
        self._weighted_losses = weighted_losses
        total_weight = sum([w for w, unused_loss in self._weighted_losses])
        self._weighted_losses = [(w / total_weight, loss)
                                 for w, loss in self._weighted_losses]

    def call(self, y_true, y_pred):
        return tf.math.add_n([weight * loss(y_true, y_pred)
                              for weight, loss in self._weighted_losses])


class TileMetric(tf.keras.metrics.Metric):
        def __init__(self, k=2, name='tile_metric', **kwargs):
            super(TileMetric, self).__init__(name=name, **kwargs)
            self.k = k
            self.tile_metric = 0

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true_min = tf.reduce_min(y_true, axis=1)
            topk_indices = tf.math.top_k(-y_pred, k=self.k).indices
            gathered_values = tf.gather(y_true, topk_indices, batch_dims=-1)
            y_pred_min = tf.reduce_min(gathered_values, axis=1)
            self.tile_metric = 2 - tf.reduce_mean(y_pred_min/y_true_min)

        def result(self):
            return self.tile_metric
