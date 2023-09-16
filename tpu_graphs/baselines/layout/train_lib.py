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

import gzip
import io
import json
import os
from typing import Any
import tqdm
import random
import numpy as np
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_ranking as tfr
from tpu_graphs.baselines.layout import data
from tpu_graphs.baselines.layout import models
from tpu_graphs.baselines.layout import train_args
from tpu_graphs.baselines.layout import metrics
from scipy import stats

_DATA_ROOT = flags.DEFINE_string(
    'data_root', '/local_datasets/tpugraphs/npz/layout',
    'Root directory containing dataset. It must contain subdirectories '
    '{train, test, valid}, each having many .npz files')
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir', '/local_datasets/tpugraphs/cache',
    'If given, dataset tensors will be cached here for faster loading. Files '
    'with name "<hash>.npz" will be written, where <hash> is a hash of the '
    'filepattern of training data, i.e., it depends on the collection e.g., '
    '{xla:default} and partition {train, test, valid}.')
_PDB = flags.DEFINE_integer(
    'debug', -1, 'If >0, pdb debugger will be entered after this many epochs.')


def _graph_and_label(graph: tfgnn.GraphTensor):
    # Return runtimes divided over large number: only ranking is required. The
    # runtimes are in the 100K range
    label = tf.cast(graph.node_sets['g']['runtimes'], tf.float32) / 1e7
    return graph, label


def save_run(run_info: dict[str, Any], out_dir: str, args: train_args.TrainArgs):
    # Save run file.
    out_run_file = os.path.join(out_dir, f'run_{args.ver}_{args.source}_{args.search}.jsonz')
    bytes_io = io.BytesIO()
    with gzip.open(bytes_io, 'wb') as fout:
        fout.write(json.dumps(run_info).encode())
    with tf.io.gfile.GFile(out_run_file, 'wb') as fout:
        fout.write(bytes_io.getvalue())
    logging.info('wrote %s', out_run_file)


_INFERENCE_CONFIGS_BATCH_SIZE = 10  # For producing inference csv, post-train.


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def train_origin(args: train_args.TrainArgs):

    # Set seed
    set_seed(args.seed)

    """Training loop. `train_args.py` contains description of arguments."""
    out_dir = os.path.expanduser(os.path.join(args.out_dir,args.ver))
    if not tf.io.gfile.exists(out_dir):
        tf.io.gfile.makedirs(out_dir)

    run_info = dict(train_curve=dict(), final_opa=dict(), final_error=dict(), args=args._asdict(),)

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------
    # Input training data.
    data_root_dir = os.path.join(os.path.expanduser(_DATA_ROOT.value), args.source, args.search)
    num_configs = args.configs
    dataset_partitions = data.get_npz_dataset(
        data_root_dir, min_train_configs=num_configs,
        max_train_configs=args.max_configs)
        #cache_dir=os.path.expanduser(_CACHE_DIR.value))
    batch_size = args.batch_size

    train_ds = (dataset_partitions.train.get_graph_tensors_dataset(num_configs, max_nodes=args.keep_nodes)
                .shuffle(100, reshuffle_each_iteration=True)
                .batch(batch_size, drop_remainder=False)
                .map(tfgnn.GraphTensor.merge_batch_to_components)
                .map(_graph_and_label))

    valid_ds = (dataset_partitions.validation.get_graph_tensors_dataset(num_configs)
                .batch(batch_size, drop_remainder=False)
                .map(tfgnn.GraphTensor.merge_batch_to_components)
                .map(_graph_and_label))

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------
    model = models.ResModel(num_configs, dataset_partitions.num_ops)
    loss = tfr.keras.losses.ListMLELoss()  # (temperature=10)
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.clip_norm)
    model.compile(loss=loss, optimizer=opt, metrics=[tfr.keras.metrics.OPAMetric(name='metric')])

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------
    train_curve = run_info['train_curve']  # For short.
    checkpoint_filepath = os.path.join(out_dir, "checkpoint_"+args.ver+'_'+args.source+'_'+args.search, "checkpoint")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                                   monitor='val_metric', mode='max',
                                                                   save_best_only=True)

    history = model.fit(train_ds, epochs=args.epochs, verbose=1, validation_data=valid_ds, callbacks=[model_checkpoint_callback])

    train_curve['loss'] = history.history['loss']
    train_curve['metric'] = history.history['metric']
    train_curve['val_loss'] = history.history['val_loss']
    train_curve['val_metric'] = history.history['val_metric']


    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------
    best_val_metric = np.max(train_curve['val_metric'])
    run_info['final_error']['val_metric'] = best_val_metric

    model.load_weights(checkpoint_filepath)
    run_info['final_error']['val'], run_info['final_error']['acc_metric'] \
        = metrics.top_error_performance(dataset_partitions.validation.get_graph_tensors_dataset(num_configs), model.forward)
    print("OPA={:0.3f},LM={:0.3f},TE={}".format(best_val_metric,run_info['final_error']['acc_metric'],run_info['final_error']['val']))
    save_run(run_info, out_dir, args)

    # --------------------------------------------------------------
    # Testing
    # --------------------------------------------------------------
    print('\n\n   Running inference on test set ...\n\n')
    assert dataset_partitions.test.graph_id is not None
    test_rankings = []
    for graph in tqdm.tqdm(dataset_partitions.test.iter_graph_tensors(),
                           total=dataset_partitions.test.graph_id.shape[-1],
                           desc='Inference'):
        num_configs = graph.node_sets['g']['runtimes'].shape[-1]
        all_scores = []
        for i in range(0, num_configs, _INFERENCE_CONFIGS_BATCH_SIZE):
            end_i = min(i + _INFERENCE_CONFIGS_BATCH_SIZE, num_configs)
            # Take a cut of the configs.
            node_set_g = graph.node_sets['g']
            subconfigs_graph = tfgnn.GraphTensor.from_pieces(
                edge_sets=graph.edge_sets,
                node_sets={
                    'op': graph.node_sets['op'],
                    'nconfig': tfgnn.NodeSet.from_fields(
                        sizes=graph.node_sets['nconfig'].sizes,
                        features={
                            'feats': graph.node_sets['nconfig']['feats'][:, i:end_i],
                        }),
                    'g': tfgnn.NodeSet.from_fields(
                        sizes=tf.constant([1]),
                        features={
                            'graph_id': node_set_g['graph_id'],
                            'runtimes': node_set_g['runtimes'][:, i:end_i],
                            'kept_node_ratio': node_set_g['kept_node_ratio'],
                        })
                })
            h = model.forward(subconfigs_graph, num_configs=(end_i - i), backprop=False)
            all_scores.append(h[0])
        all_scores = tf.concat(all_scores, axis=0)
        graph_id = graph.node_sets['g']['graph_id'][0].numpy().decode()
        sorted_indices = tf.strings.join(tf.strings.as_string(tf.argsort(all_scores)), ';').numpy().decode()
        test_rankings.append((graph_id, sorted_indices))

    with tf.io.gfile.GFile(args.results_csv, 'w') as fout:
        fout.write('ID,TopConfigs\n')
        for graph_id, ranks in test_rankings:
            fout.write(f'layout:{args.source}:{args.search}:{graph_id},{ranks}\n')
    print('\n\n   ***  Wrote', args.results_csv, '\n\n')


def train(args: train_args.TrainArgs):

    # Set seed
    set_seed(args.seed)

    """Training loop. `train_args.py` contains description of arguments."""
    out_dir = os.path.expanduser(os.path.join(args.out_dir,args.ver))
    if not tf.io.gfile.exists(out_dir):
        tf.io.gfile.makedirs(out_dir)

    run_info = dict(train_curve=dict(), final_opa=dict(), final_error=dict(), args=args._asdict(),)

    for rep in range(args.reps):
        print("----> Starting training prodecure #{}".format(rep+1))

        # --------------------------------------------------------------
        # Dataset
        # --------------------------------------------------------------
        # Input training data.
        data_root_dir = os.path.join(os.path.expanduser(_DATA_ROOT.value), args.source, args.search)
        num_configs = args.configs
        dataset_partitions = data.get_npz_dataset(
            data_root_dir, min_train_configs=num_configs,
            max_train_configs=args.max_configs, rep=rep)
            #cache_dir=os.path.expanduser(_CACHE_DIR.value))
        batch_size = args.batch_size

        train_ds = (dataset_partitions.train.get_graph_tensors_dataset(num_configs, max_nodes=args.keep_nodes)
                    .shuffle(100, reshuffle_each_iteration=True)
                    .batch(batch_size, drop_remainder=False)
                    .map(tfgnn.GraphTensor.merge_batch_to_components)
                    .map(_graph_and_label))

        valid_ds = (dataset_partitions.validation.get_graph_tensors_dataset(num_configs)
                    .batch(batch_size, drop_remainder=False)
                    .map(tfgnn.GraphTensor.merge_batch_to_components)
                    .map(_graph_and_label))

        # --------------------------------------------------------------
        # Model
        # --------------------------------------------------------------
        model = models.ResModel(num_configs, dataset_partitions.num_ops)
        #loss = metrics.CombinedLoss(metrics.parse_loss_str(args.losses))
        #loss = tfr.keras.losses.ListMLELoss()
        loss = tfr.keras.losses.PairwiseHingeLoss()
        learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.learning_rate,
                                                                          decay_steps=1000, decay_rate=0.96)
        #opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, clipnorm=args.clip_norm)
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.clip_norm)
        model.compile(loss=loss, optimizer=opt, metrics=[tfr.keras.metrics.OPAMetric(name='metric')])

        # --------------------------------------------------------------
        # Training
        # --------------------------------------------------------------
        train_curve = run_info['train_curve']
        checkpoint_filepath = os.path.join(out_dir, "checkpoint_"+args.ver+'_'+args.source+'_'+args.search, "checkpoint")
        if os.path.exists(checkpoint_filepath):
            print("----> Loading weights from {}".format(checkpoint_filepath))
            model.load_weights(checkpoint_filepath) # load from checkpoint
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                                       monitor='val_metric', mode='max',
                                                                       save_best_only=True)

        history = model.fit(train_ds, epochs=args.epochs, verbose=1, validation_data=valid_ds, callbacks=[model_checkpoint_callback])

        train_curve['loss'] = history.history['loss']
        train_curve['metric'] = history.history['metric']
        train_curve['val_loss'] = history.history['val_loss']
        train_curve['val_metric'] = history.history['val_metric']
        del train_ds, valid_ds

    # Save best OPA
    best_val_metric = np.max(train_curve['val_metric'])
    run_info['final_error']['val_metric'] = best_val_metric
    model.load_weights(checkpoint_filepath)

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------
    print('\n\n   Running inference on valid set ...\n\n')
    assert dataset_partitions.validation.graph_id is not None
    valid_rankings = []
    for graph in tqdm.tqdm(dataset_partitions.validation.iter_graph_tensors(), total=dataset_partitions.validation.graph_id.shape[-1], desc='Inference'):
        num_configs = graph.node_sets['g']['runtimes'].shape[-1]
        all_scores = []
        runtimes = tf.squeeze(graph.node_sets['g']['runtimes'])
        for i in range(0, num_configs, _INFERENCE_CONFIGS_BATCH_SIZE):
            end_i = min(i + _INFERENCE_CONFIGS_BATCH_SIZE, num_configs)
            # Take a cut of the configs.
            node_set_g = graph.node_sets['g']
            subconfigs_graph = tfgnn.GraphTensor.from_pieces(
                edge_sets=graph.edge_sets,
                node_sets={
                    'op': graph.node_sets['op'],
                    'nconfig': tfgnn.NodeSet.from_fields(
                        sizes=graph.node_sets['nconfig'].sizes,
                        features={
                            'feats': graph.node_sets['nconfig']['feats'][:, i:end_i],
                        }),
                    'g': tfgnn.NodeSet.from_fields(
                        sizes=tf.constant([1]),
                        features={
                            'graph_id': node_set_g['graph_id'],
                            'runtimes': node_set_g['runtimes'][:, i:end_i],
                            'kept_node_ratio': node_set_g['kept_node_ratio'],
                        })
                })
            h = model.forward(subconfigs_graph, num_configs=(end_i - i), backprop=False)
            all_scores.append(h[0])
        all_scores = tf.concat(all_scores, axis=0)
        ktau = stats.kendalltau(all_scores.numpy(), runtimes.numpy()).statistic
        valid_rankings.append(ktau)

    final_ktau = np.mean(np.array(valid_rankings))
    run_info['final_error']['acc_metric'] = final_ktau
    print("----> OPA={:0.3f},KTAU={:0.3f}".format(best_val_metric,final_ktau))
    save_run(run_info, out_dir, args)

    # --------------------------------------------------------------
    # Testing
    # --------------------------------------------------------------
    print('\n\n   Running inference on test set ...\n\n')
    assert dataset_partitions.test.graph_id is not None
    test_rankings = []
    for graph in tqdm.tqdm(dataset_partitions.test.iter_graph_tensors(), total=dataset_partitions.test.graph_id.shape[-1], desc='Inference'):
        num_configs = graph.node_sets['g']['runtimes'].shape[-1]
        all_scores = []
        for i in range(0, num_configs, _INFERENCE_CONFIGS_BATCH_SIZE):
            end_i = min(i + _INFERENCE_CONFIGS_BATCH_SIZE, num_configs)
            # Take a cut of the configs.
            node_set_g = graph.node_sets['g']
            subconfigs_graph = tfgnn.GraphTensor.from_pieces(
                edge_sets=graph.edge_sets,
                node_sets={
                    'op': graph.node_sets['op'],
                    'nconfig': tfgnn.NodeSet.from_fields(
                        sizes=graph.node_sets['nconfig'].sizes,
                        features={
                            'feats': graph.node_sets['nconfig']['feats'][:, i:end_i],
                        }),
                    'g': tfgnn.NodeSet.from_fields(
                        sizes=tf.constant([1]),
                        features={
                            'graph_id': node_set_g['graph_id'],
                            'runtimes': node_set_g['runtimes'][:, i:end_i],
                            'kept_node_ratio': node_set_g['kept_node_ratio'],
                        })
                })
            h = model.forward(subconfigs_graph, num_configs=(end_i - i), backprop=False)
            all_scores.append(h[0])
        all_scores = tf.concat(all_scores, axis=0)
        graph_id = graph.node_sets['g']['graph_id'][0].numpy().decode()
        sorted_indices = tf.strings.join(tf.strings.as_string(tf.argsort(all_scores)), ';').numpy().decode()
        test_rankings.append((graph_id, sorted_indices))

    with tf.io.gfile.GFile(args.results_csv, 'w') as fout:
        fout.write('ID,TopConfigs\n')
        for graph_id, ranks in test_rankings:
            fout.write(f'layout:{args.source}:{args.search}:{graph_id},{ranks}\n')
    print('\n\n   ***  Wrote', args.results_csv, '\n\n')


def evaluate(args: train_args.TrainArgs):

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------
    # Input training data.
    data_root_dir = os.path.join(os.path.expanduser(_DATA_ROOT.value), args.source, args.search)
    num_configs = args.configs
    dataset_valid = data.get_npz_valid_dataset(
        data_root_dir, min_train_configs=num_configs,
        max_train_configs=args.max_configs)
        #cache_dir=os.path.expanduser(_CACHE_DIR.value))

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------
    model = models.ResModel(num_configs, 119) #dataset_valid.num_ops)
    out_dir = os.path.expanduser(os.path.join(args.out_dir,args.ver))
    checkpoint_filepath = os.path.join(out_dir, "checkpoint_"+args.ver+'_'+args.source+'_'+args.search, "checkpoint")
    model.compile(run_eagerly=True)

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------
    print('\n\n   Running inference on valid set ...\n\n')
    model.load_weights(checkpoint_filepath)
    assert dataset_valid.graph_id is not None
    valid_rankings = []
    for graph in tqdm.tqdm(dataset_valid.iter_graph_tensors(), total=dataset_valid.graph_id.shape[-1], desc='Inference'):
        num_configs = graph.node_sets['g']['runtimes'].shape[-1]
        all_scores = []
        runtimes = tf.squeeze(graph.node_sets['g']['runtimes'])
        for i in range(0, num_configs, _INFERENCE_CONFIGS_BATCH_SIZE):
            end_i = min(i + _INFERENCE_CONFIGS_BATCH_SIZE, num_configs)
            # Take a cut of the configs.
            node_set_g = graph.node_sets['g']
            subconfigs_graph = tfgnn.GraphTensor.from_pieces(
                edge_sets=graph.edge_sets,
                node_sets={
                    'op': graph.node_sets['op'],
                    'nconfig': tfgnn.NodeSet.from_fields(
                        sizes=graph.node_sets['nconfig'].sizes,
                        features={
                            'feats': graph.node_sets['nconfig']['feats'][:, i:end_i],
                        }),
                    'g': tfgnn.NodeSet.from_fields(
                        sizes=tf.constant([1]),
                        features={
                            'graph_id': node_set_g['graph_id'],
                            'runtimes': node_set_g['runtimes'][:, i:end_i],
                            'kept_node_ratio': node_set_g['kept_node_ratio'],
                        })
                })
            h = model.forward(subconfigs_graph, num_configs=(end_i - i), backprop=False)
            all_scores.append(h[0])
        all_scores = tf.concat(all_scores, axis=0)
        ktau = stats.kendalltau(all_scores.numpy(), runtimes.numpy()).statistic
        #if ktau < 0.2:
        id = graph.node_sets['g']['tile_id'][0]
        print(f'{id}:{ktau}')
        valid_rankings.append(ktau)

    final_ktau = np.mean(np.array(valid_rankings))
    print("KTAU={:0.3f}".format(final_ktau))


def test(args: train_args.TrainArgs):


    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------
    # Input training data.
    data_root_dir = os.path.join(os.path.expanduser(_DATA_ROOT.value), args.source, args.search)
    num_configs = args.configs
    dataset_partitions = data.get_npz_dataset(
        data_root_dir, min_train_configs=num_configs,
        max_train_configs=args.max_configs)
        #cache_dir=os.path.expanduser(_CACHE_DIR.value))

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------
    model = models.ResModel(num_configs, dataset_partitions.num_ops)
    out_dir = os.path.expanduser(os.path.join(args.out_dir,args.ver))
    checkpoint_filepath = os.path.join(out_dir, "checkpoint_"+args.ver+'_'+args.source+'_'+args.search, "checkpoint")

    # --------------------------------------------------------------
    # Testing
    # --------------------------------------------------------------
    print('\n\n   Running inference on test set ...\n\n')
    model.load_weights(checkpoint_filepath)
    assert dataset_partitions.test.graph_id is not None
    test_rankings = []
    for graph in tqdm.tqdm(dataset_partitions.test.iter_graph_tensors(),
                           total=dataset_partitions.test.graph_id.shape[-1],
                           desc='Inference'):
        num_configs = graph.node_sets['g']['runtimes'].shape[-1]
        all_scores = []
        for i in range(0, num_configs, _INFERENCE_CONFIGS_BATCH_SIZE):
            end_i = min(i + _INFERENCE_CONFIGS_BATCH_SIZE, num_configs)
            # Take a cut of the configs.
            node_set_g = graph.node_sets['g']
            subconfigs_graph = tfgnn.GraphTensor.from_pieces(
                edge_sets=graph.edge_sets,
                node_sets={
                    'op': graph.node_sets['op'],
                    'nconfig': tfgnn.NodeSet.from_fields(
                        sizes=graph.node_sets['nconfig'].sizes,
                        features={
                            'feats': graph.node_sets['nconfig']['feats'][:, i:end_i],
                        }),
                    'g': tfgnn.NodeSet.from_fields(
                        sizes=tf.constant([1]),
                        features={
                            'graph_id': node_set_g['graph_id'],
                            'runtimes': node_set_g['runtimes'][:, i:end_i],
                            'kept_node_ratio': node_set_g['kept_node_ratio'],
                        })
                })
            h = model.forward(subconfigs_graph, num_configs=(end_i - i), backprop=False)
            all_scores.append(h[0])
        all_scores = tf.concat(all_scores, axis=0)
        graph_id = graph.node_sets['g']['graph_id'][0].numpy().decode()
        sorted_indices = tf.strings.join(tf.strings.as_string(tf.argsort(all_scores)), ';').numpy().decode()
        test_rankings.append((graph_id, sorted_indices))

    with tf.io.gfile.GFile(args.results_csv, 'w') as fout:
        fout.write('ID,TopConfigs\n')
        for graph_id, ranks in test_rankings:
            fout.write(f'layout:{args.source}:{args.search}:{graph_id},{ranks}\n')
    print('\n\n   ***  Wrote', args.results_csv, '\n\n')
