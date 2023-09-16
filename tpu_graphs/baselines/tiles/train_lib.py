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

"""Library for running train-and-eval loop on tiles dataset."""

import functools
import gzip
import io
import json
import os
from typing import Callable, Any
import numpy as np
import random
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_ranking as tfr
from tpu_graphs.baselines.tiles import data
from tpu_graphs.baselines.tiles import metrics
from tpu_graphs.baselines.tiles import models
from tpu_graphs.baselines.tiles import train_args
import tqdm
from sklearn.model_selection import KFold

_DATA_ROOT = flags.DEFINE_string(
    'data_root', '/local_datasets/tpugraphs/npz/tile/xla',
    'Root directory containing dataset. It must contain subdirectories '
    '{train, test, validation}, each having many .npz files')
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir', '/local_datasets/tpugraphs/cache/tile',
    'If given, dataset tensors will be cached here for faster loading.')


def _graph_and_label(graph: tfgnn.GraphTensor, batch_size=10, num_configs=2):
    label = tf.reshape(
        graph.node_sets['config']['runtimes'], [batch_size, num_configs])
    return graph, label


# Used for validation. For training, data.py accepts `min_train_configs`.
def _graph_has_enough_configs(graph: tfgnn.GraphTensor, num_configs=2):
    """To used to filter validation dataset."""
    return graph.node_sets['config'].sizes[0] >= num_configs


def save_run(run_info: dict[str, Any], out_dir: str, args: train_args.TrainArgs, kf=""):
    """Writes `model` and `run_info` onto `out_dir`/*`args.compute_hash()`*."""
    # Save run file.
    if kf != "":
        out_run_file = os.path.join(out_dir, f'run_{args.ver}f{kf}_tile.jsonz')
    else:
        out_run_file = os.path.join(out_dir, f'run_{args.ver}_tile.jsonz')
    bytes_io = io.BytesIO()
    with gzip.open(bytes_io, 'wb') as fout:
        fout.write(json.dumps(run_info).encode())
    with tf.io.gfile.GFile(out_run_file, 'wb') as fout:
        fout.write(bytes_io.getvalue())
    logging.info('wrote %s', out_run_file)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def train(args: train_args.TrainArgs):

    # Set seed
    set_seed(args.seed)

    """Training loop. `train_args.py` contains description of arguments."""
    out_dir = os.path.expanduser(os.path.join(args.out_dir,args.ver))
    if not tf.io.gfile.exists(out_dir):
        tf.io.gfile.makedirs(out_dir)

    # Will be written in out_dir.
    run_info = dict(train_curve=dict(), final_opa=dict(), final_error=dict(), args=args._asdict(),)

    # -------------------------------------------
    # Dataset
    # -------------------------------------------
    data_root_dir = os.path.expanduser(_DATA_ROOT.value)
    num_configs = args.configs
    dataset_partitions = data.get_npz_dataset(
        data_root_dir, min_train_configs=num_configs)
        #cache_dir=os.path.expanduser(_CACHE_DIR.value))
    batch_size = args.batch_size
    attach_labels_fn = functools.partial(_graph_and_label, batch_size=batch_size, num_configs=num_configs)
    train_ds = (dataset_partitions.train.get_graph_tensors_dataset(num_configs)
                .shuffle(5000, reshuffle_each_iteration=True)
                .batch(batch_size, drop_remainder=True)
                .map(tfgnn.GraphTensor.merge_batch_to_components)
                .map(attach_labels_fn))

    valid_ds = (dataset_partitions.validation.get_graph_tensors_dataset(num_configs)
                # Get an extra 5% as we follow with `filter()`.
                .take(int(args.validate_batches * batch_size * 1.05))
                .filter(functools.partial(_graph_has_enough_configs, num_configs=num_configs))
                .batch(batch_size, drop_remainder=True)
                .map(tfgnn.GraphTensor.merge_batch_to_components)
                .map(attach_labels_fn))

    # -------------------------------------------
    # Model
    # -------------------------------------------
    model_class = getattr(models, args.model)
    model_kwargs = json.loads(args.model_kwargs_json)
    num_ops = dataset_partitions.num_ops
    model = model_class(num_configs, num_ops, **model_kwargs)
    loss = tfr.keras.losses.PairwiseHingeLoss()
    #loss = metrics.CombinedLoss(metrics.parse_loss_str(args.losses))
    #loss = tfr.keras.losses.ListMLELoss()
    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.learning_rate,
                                                                      decay_steps=1000, decay_rate=0.96)
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.clip_norm)
    model.compile(loss=loss, optimizer=opt, metrics=[tfr.keras.metrics.OPAMetric(name='metric')])

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------
    train_curve = run_info['train_curve']
    checkpoint_filepath = os.path.join(out_dir, "checkpoint_"+args.ver+'_tile', "checkpoint")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                                   monitor='val_metric', mode='max',
                                                                   save_best_only=True)

    history = model.fit(train_ds, epochs=args.epochs, verbose=1, validation_data=valid_ds, callbacks=[model_checkpoint_callback])

    train_curve['loss'] = history.history['loss']
    train_curve['metric'] = history.history['metric']
    train_curve['val_loss'] = history.history['val_loss']
    train_curve['val_metric'] = history.history['val_metric']

    # -------------------------------------------
    # Validation
    # -------------------------------------------
    best_val_metric = np.max(train_curve['val_metric'])
    run_info['final_error']['val_metric'] = best_val_metric
    model.load_weights(checkpoint_filepath)

    valid_ds = dataset_partitions.validation.get_graph_tensors_dataset()
    run_info['final_error']['val'], run_info['final_error']['acc_metric'] = metrics.top_error_performance(valid_ds, model.forward)
    print("OPA={:0.3f},TM={:0.3f},TE={}".format(best_val_metric, run_info['final_error']['acc_metric'],run_info['final_error']['val']))
    save_run(run_info, out_dir, args)

    # -------------------------------------------
    # Testing
    # -------------------------------------------
    test_ds = dataset_partitions.test.get_graph_tensors_dataset()
    module_ids, ranks = rank_config_indices(test_ds, model.forward)
    write_least_runtimes_csv(args.results_csv, module_ids, ranks)
    ### Add test predictions into run_info file.
    run_info['test_predictions'] = {}
    module_ids = module_ids.numpy().tolist()
    predictions = ranks.numpy().tolist()
    for module_id, module_predictions in zip(module_ids, predictions):
        module_id = bytes(module_id).decode()
        run_info['test_predictions'][module_id] = module_predictions


def test(args: train_args.TrainArgs):

    # -------------------------------------------
    # Dataset
    # -------------------------------------------
    data_root_dir = os.path.expanduser(_DATA_ROOT.value)
    num_configs = args.configs
    dataset_partitions = data.get_npz_dataset(
        data_root_dir, min_train_configs=num_configs,
        cache_dir=os.path.expanduser(_CACHE_DIR.value))

    # -------------------------------------------
    # Model
    # -------------------------------------------
    model_class = getattr(models, args.model)
    model_kwargs = json.loads(args.model_kwargs_json)
    num_ops = dataset_partitions.num_ops
    model = model_class(num_configs, num_ops, **model_kwargs)
    out_dir = os.path.expanduser(os.path.join(args.out_dir,args.ver))
    checkpoint_filepath = os.path.join(out_dir, "checkpoint_"+args.ver+'_tile', "checkpoint")
    model.load_weights(checkpoint_filepath)

    # -------------------------------------------
    # Testing
    # -------------------------------------------
    test_ds = dataset_partitions.test.get_graph_tensors_dataset()
    module_ids, ranks = rank_config_indices(test_ds, model.forward)
    results_csv = args.results_csv.split('.csv')[0] + ".csv"
    write_least_runtimes_csv(results_csv, module_ids, ranks)


def rank_config_indices(test_ds: tf.data.Dataset, model_fn: Callable[[tfgnn.GraphTensor, int], tf.Tensor], top_ranks=5) -> tuple[tf.Tensor, tf.Tensor]:
    """Module IDs and config indices that `model_fn` assigns lowest scores.

  Args:
    test_ds: Test dataset containing `GraphTensor` instances. Each instance must
      have node sets `'config'` and `'g'` (with feature 'tile_id')
    model_fn: Callable (e.g., tf.Keras model) that will be invoked on every item
      in `test_ds` and the number of configurations (=N). It is expeted to
      return tensor of shape (1, N). The least indices will be output.
    top_ranks: Only this many least indices will be kept.

  Returns:
    Two `tf.Tensor` instances. The first is a vector with entry `[i]` being the
    `graph.node_sets['g']['tile_id']` of the `i`th element of `test_ds`. The
    second is a matrix with width `top_ranks`, where row `[i]` being the least
    `top_ranks` indices when invoking `model_fn` on `graph`.
  """
    all_sorted_indices = []
    all_module_ids = []
    for graph in tqdm.tqdm(test_ds, desc='Generating Predictions'):
        num_configs = int(graph.node_sets['config'].sizes[0].numpy())
        preds = model_fn(graph, num_configs)
        preds = tf.squeeze(preds, 0)  # Remove batch size (of 1)
        sorted_indices = tf.argsort(preds)
        sorted_indices = tf.concat([sorted_indices, tf.zeros([top_ranks], dtype=sorted_indices.dtype)], axis=0)
        sorted_indices = sorted_indices[:top_ranks]
        all_sorted_indices.append(sorted_indices)
        all_module_ids.append(graph.node_sets['g']['tile_id'][0])

    return tf.stack(all_module_ids, axis=0), tf.stack(all_sorted_indices, axis=0)


def write_least_runtimes_csv(out_csv_filepath: str, module_ids: tf.Tensor, ranks: tf.Tensor):
    """Writes CSV file with line `i` containing module_ids[i] and ranks[i]."""
    csv_ranks = tf.strings.join(tf.strings.as_string(tf.transpose(ranks)), ';')

    stack_join = lambda x, delim: tf.strings.join(tf.stack(x), delim)
    with tf.io.gfile.GFile(out_csv_filepath, 'w') as fout:
        fout.write('ID,TopConfigs\n')
        id_vector = stack_join([tf.fill(module_ids.shape, 'tile:xla'), module_ids], ':')
        csv_lines = stack_join([id_vector, csv_ranks], ',')
        fout.write(stack_join(csv_lines, '\n').numpy().decode('utf-8'))
    print('\n\n   ***  Wrote', out_csv_filepath, '\n\n')
