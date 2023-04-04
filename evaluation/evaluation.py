import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


def eval_edge_prediction(model, negative_edge_sampler, data_1,data_2, n_neighbors, batch_size=500):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_acc, val_ap, val_auc = [], [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data_1.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch_1 = data_1.sources[s_idx:e_idx]
      destinations_batch_1 = data_1.destinations[s_idx:e_idx]
      timestamps_batch_1 = data_1.timestamps[s_idx:e_idx]
      edge_idxs_batch_1 = data_1.edge_idxs[s_idx: e_idx]

      sources_batch_2 = data_2.sources[s_idx:e_idx]
      destinations_batch_2 = data_2.destinations[s_idx:e_idx]
      timestamps_batch_2 = data_2.timestamps[s_idx:e_idx]
      edge_idxs_batch_2 = data_2.edge_idxs[s_idx: e_idx]


      size = len(sources_batch_1)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch_1, destinations_batch_1,
                                                            negative_samples, timestamps_batch_1,
                                                            edge_idxs_batch_1,sources_batch_2,
                                                            destinations_batch_2,timestamps_batch_2,
                                                            edge_idxs_batch_2, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      pred_label = [1 if prob >= 0.5 else 0 for prob in pred_score]
      val_acc.append(accuracy_score(true_label, pred_label))
      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))
      

  return np.mean(val_acc), np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(tgn, decoder, data_1, edge_idxs_1, data_2, edge_idxs_2, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data_1.sources))
  num_instance = len(data_1.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch_1 = data_1.sources[s_idx: e_idx]
      destinations_batch_1 = data_1.destinations[s_idx: e_idx]
      timestamps_batch_1 = data_1.timestamps[s_idx:e_idx]
      edge_idxs_batch_1 = edge_idxs_1[s_idx: e_idx]

      sources_batch_2 = data_2.sources[s_idx: e_idx]
      destinations_batch_2 = data_2.destinations[s_idx: e_idx]
      timestamps_batch_2 = data_2.timestamps[s_idx:e_idx]
      edge_idxs_batch_2 = edge_idxs_2[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch_1,
                                                                                   destinations_batch_1,
                                                                                   destinations_batch_1,
                                                                                   timestamps_batch_1,
                                                                                   edge_idxs_batch_1,
                                                                                   sources_batch_2,
                                                                                   destinations_batch_2,
                                                                                   timestamps_batch_2,
                                                                                   edge_idxs_batch_2,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data_1.labels, pred_prob)
  return auc_roc
