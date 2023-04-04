import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features_1, edge_features_1, node_features_2, edge_features_2,device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    self.node_raw_features_1 = torch.from_numpy(node_features_1.astype(np.float32)).to(device)
    self.edge_raw_features_1 = torch.from_numpy(edge_features_1.astype(np.float32)).to(device)

    self.node_raw_features_2 = torch.from_numpy(node_features_2.astype(np.float32)).to(device)
    self.edge_raw_features_2 = torch.from_numpy(edge_features_2.astype(np.float32)).to(device)

    self.n_node_features = self.node_raw_features_1.shape[1]
    self.n_nodes = self.node_raw_features_1.shape[0]
    self.n_edge_features = self.edge_raw_features_1.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep

    self.use_memory = use_memory
    self.time_encoder_1 = TimeEncode(dimension=self.n_node_features)
    self.time_encoder_2 = TimeEncode(dimension=self.n_node_features)
    self.memory_1 = None
    self.memory_2 = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    if self.use_memory:
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder_1.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.memory_1 = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.memory_2 = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater_1 = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory_1,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)
      self.memory_updater_2 = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory_2,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features_1,
                                                 edge_features=self.edge_raw_features_1,
                                                 memory=self.memory_1,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder_1,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors)

    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                     self.n_node_features,
                                     1)


    self.transition_cnt = torch.zeros(self.n_nodes).long()

  def compute_temporal_embeddings(self, source_nodes_1, destination_nodes_1, negative_nodes_1, edge_times_1,
                                  edge_idxs_1, source_nodes_2,destination_nodes_2, edge_times_2,
                                 edge_idxs_2, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    n_samples = len(source_nodes_1)
    nodes = np.concatenate([source_nodes_1, destination_nodes_1, negative_nodes_1])
    positives = np.concatenate([source_nodes_1, destination_nodes_1])
    timestamps = np.concatenate([edge_times_1, edge_times_1, edge_times_1])

    nodes_2 = np.concatenate([source_nodes_2, destination_nodes_2])

    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches

        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory_1.messages)
      else:
        memory = self.memory_1.get_memory(list(range(self.n_nodes)))
        last_update = self.memory_1.last_update

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times_1).to(self.device) - last_update[
        source_nodes_1].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times_1).to(self.device) - last_update[
        destination_nodes_1].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times_1).to(self.device) - last_update[
        negative_nodes_1].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0)

    # Compute the embeddings using the embedding module
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        self.update_memory(positives, self.memory_1.messages,self.memory_updater_1)

        self.transition_cnt[positives] = self.transition_cnt[positives] + 1

        assert torch.allclose(memory[positives], self.memory_1.get_memory(positives), atol=1e-5), \
          "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory_1.clear_messages(positives)

        self.update_memory(nodes_2, self.memory_2.messages,self.memory_updater_2)
        self.memory_2.clear_messages(nodes_2)

        for i in range(self.n_nodes):
          if self.transition_cnt[i] % 10 == 0:
            state = self.memory_2.get_memory(i)
            self.memory_1.set_memory(i,state)


        self.transition_cnt = torch.zeros(self.n_nodes).long()
        self.update_memory(positives, self.memory_1.messages,self.memory_updater_1)
        self.transition_cnt[positives] = self.transition_cnt[positives] + 1
        self.memory_1.clear_messages(positives)

        self.update_memory(nodes_2, self.memory_2.messages,self.memory_updater_2)
        self.memory_2.clear_messages(nodes_2)

        for i in range(self.n_nodes):
          if self.transition_cnt[i] % 10 == 0:
            state = self.memory_2.get_memory(i)
            self.memory_1.set_memory(i,state)



      unique_sources_1, source_id_to_messages_1 = self.get_raw_messages_1(source_nodes_1,
                                                                    source_node_embedding,
                                                                    destination_nodes_1,
                                                                    destination_node_embedding,
                                                                    edge_times_1, edge_idxs_1)
      unique_destinations_1, destination_id_to_messages_1 = self.get_raw_messages_1(destination_nodes_1,
                                                                              destination_node_embedding,
                                                                              source_nodes_1,
                                                                              source_node_embedding,
                                                                              edge_times_1, edge_idxs_1)


      unique_sources_2, source_id_to_messages_2 = self.get_raw_messages_2(source_nodes_2,
                                                                    None,
                                                                    destination_nodes_2,
                                                                    None,
                                                                    edge_times_2, edge_idxs_2)
      unique_destinations_2, destination_id_to_messages_2 = self.get_raw_messages_2(destination_nodes_2,
                                                                              None,
                                                                              source_nodes_2,
                                                                              None,
                                                                              edge_times_2, edge_idxs_2)
      if self.memory_update_at_start:
        self.memory_1.store_raw_messages(unique_sources_1, source_id_to_messages_1)
        self.memory_1.store_raw_messages(unique_destinations_1, destination_id_to_messages_1)

        self.memory_2.store_raw_messages(unique_sources_2, source_id_to_messages_2)
        self.memory_2.store_raw_messages(unique_destinations_2, destination_id_to_messages_2)
      else:
        self.update_memory(unique_sources_1, source_id_to_messages_1)
        self.update_memory(unique_destinations_1, destination_id_to_messages_1)
        self.update_memory(unique_sources_2, source_id_to_messages_2)
        self.update_memory(unique_destinations_2, destination_id_to_messages_2)

      if self.dyrep:
        source_node_embedding = memory[source_nodes_1]
        destination_node_embedding = memory[destination_nodes_1]
        negative_node_embedding = memory[negative_nodes_1]

    return source_node_embedding, destination_node_embedding, negative_node_embedding


  def compute_edge_probabilities(self, source_nodes_1, destination_nodes_1, negative_nodes_1, edge_times_1,
                                 edge_idxs_1,source_nodes_2,destination_nodes_2, edge_times_2,
                                 edge_idxs_2, n_neighbors=20):
    n_samples = len(source_nodes_1)
    source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
      source_nodes_1, destination_nodes_1, negative_nodes_1, edge_times_1, edge_idxs_1,source_nodes_2,destination_nodes_2, edge_times_2,
                                 edge_idxs_2, n_neighbors)

    score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding,
                                           negative_node_embedding])).squeeze(dim=0)
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]

    return pos_score.sigmoid(), neg_score.sigmoid()

  def update_memory(self, nodes, messages,memory_updater):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_memory, updated_last_update = self.memory_updater_1.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def get_raw_messages_1(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features_1[edge_idxs]

    source_memory = self.memory_2.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory_2.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory_1.last_update[source_nodes]


    source_time_delta_encoding = self.time_encoder_1(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def get_raw_messages_2(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features_2[edge_idxs]

    source_memory = self.memory_2.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory_2.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory_2.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder_2(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
