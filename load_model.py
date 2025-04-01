
from torch import nn
import torch
import pandas as pd



N_CLASSES = 2
N_NEURONS = 15
DATA_SIZE = 2048

class BagModel(nn.Module):

    def __init__(self, prepNN, afterNN, aggregation_func):
        super().__init__()

        self.prepNN = prepNN
        self.aggregation_func = aggregation_func
        self.afterNN = afterNN

    def forward(self, input):

        if not isinstance(input, tuple):
            input = (input, torch.zeros(input.size()[0]))

        ids = input[1]
        input = input[0]

        # Modify shape of bagids if only 1d tensor
        if (len(ids.shape) == 1):
            ids.resize_(1, len(ids))

        inner_ids = ids[len(ids) - 1]

        device = input.device

        NN_out = self.prepNN(input)

        unique, inverse, counts = torch.unique(inner_ids, sorted=True, return_inverse=True, return_counts=True)
        idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
        bags = unique[idx]
        counts = counts[idx]

        output = torch.empty((len(bags), len(NN_out[0])), device=device)

        for i, bag in enumerate(bags):
            output[i] = self.aggregation_func(NN_out[inner_ids == bag], dim=0)

        output = self.afterNN(output)

        if (ids.shape[0] == 1):
            return output
        else:
            ids = ids[:len(ids) - 1]
            mask = torch.empty(0, device=device).long()
            for i in range(len(counts)):
                mask = torch.cat((mask, torch.sum(counts[:i], dtype=torch.int64).reshape(1)))
            return (output, ids[:, mask])

    def _calc_mse_(self, input, labels=None):

        if not isinstance(input, tuple):
            input = (input, torch.zeros(input.size()[0]))

        calc_mse = lambda x, y: np.mean((x - y) ** 2)
        p_value, mse_value = [], []
        for row in range(0, input[0].size()[0]):
            i_input = (input[0][row:row + 1, :], torch.unsqueeze(torch.tensor(row), dim=0))
            i_p = self.forward(i_input)
            i_p = i_p.detach().cpu()

            if labels is None:
                t_label = torch.argmax(i_p, dim=1)
                t_label = int(t_label.numpy())
            else:
                t_label = int(labels.numpy())
                # t_label = int(labels[idx].cpu().numpy())

            i_p = i_p.numpy()
            n_classes = i_p.shape[1]

            one_hot = np.zeros(n_classes)
            one_hot[t_label] = 1

            p_value.append(i_p)
            mse_value.append(calc_mse(i_p, one_hot))

        return p_value, mse_value

    def mse(self, input, labels=None, bagids=None):

        if bagids is None:
            p_value, mse_value = self._calc_mse_(input, labels)
            return p_value, mse_value
        else:
            bagids = bagids.squeeze()
            p_value, mse_value, ids = [], [], []
            for i, id in enumerate(list(np.unique(bagids))):
                p, mse = self._calc_mse_(input[bagids == id], labels=labels[i])
                p_value.append(p)
                mse_value.append(mse)
                ids.append(bagids[bagids.squeeze() == id])

            p_value = np.concatenate(p_value, axis = 0 )
            mse_value = np.concatenate(mse_value, axis=0)
            ids = torch.cat(ids, dim = 0)

        return p_value, mse_value, ids

    def get_decision_df(self, input, labels=None, file_list=None):

        p_value, mse_value = self.mse(input, labels=labels)

        if file_list is None:
            df = pd.DataFrame({'p': p_value, 'mse': mse_value})
        else:
            df = pd.DataFrame({'p': p_value, 'mse': mse_value, 'files': file_list})
        df = df.sort_values(by='mse')

        return df

    def get_most_important(self, input, labels=None, file_list=None, tresh=0.2):

        df = self.get_decision_df(input, labels=labels, file_list=file_list)

        idx = df.mse <= tresh
        vip_list = df.files[idx]

        return list(vip_list)

class Aggregation(nn.Module):
    def __init__(self, linear_nodes = 15, attention_nodes = 15, dim = 0, aggregation_func = None):
        super().__init__()
        self.linear_nodes = linear_nodes
        self.attention_nodes = attention_nodes
        self.dim = dim
        self.aggregation_func = aggregation_func

        self.attention_layer = nn.Sequential(
            nn.Linear(self.linear_nodes, self.attention_nodes),
            nn.Tanh(),
            nn.Linear(self.attention_nodes,1)
        )

    def forward(self, x, dim = None):
        gate = self.attention_layer(x)
        attention_map= x*gate
        if dim is None:
            dim = self.dim

        if self.aggregation_func is None:
            attention = torch.mean(attention_map, dim = dim)
        else:
            attention = self.aggregation_func(attention_map, dim=dim)

        return attention
def get_model():

    prepNN = torch.nn.Sequential(
      torch.nn.Linear(DATA_SIZE, N_NEURONS),
      torch.nn.ReLU(),
    )

    #from mil.aggregation_layer import Aggregation
    agg_func = Aggregation(aggregation_func = torch.mean,
                           linear_nodes=N_NEURONS,
                           attention_nodes=N_NEURONS)

    afterNN = torch.nn.Sequential(
      torch.nn.Dropout(0.25),
      torch.nn.Linear(N_NEURONS, N_CLASSES),
      torch.nn.Softmax(dim = 1))

    model = BagModel(prepNN, afterNN, agg_func)

    return model

#main function
if __name__ == "__main__":

    #set a new model and load the trained weights
    model = get_model()
    pretrained_weights = torch.load("PretrainedModel#1.pth", map_location='cpu')
    model.load_state_dict(pretrained_weights)

    # load test batch
    test_bag = torch.load("test_bag#1.pt")

    #test model on test bag
    test_probs = model(test_bag)
