class TextDataset(torch.utils.data.Dataset):
    def __init__(self, lines, batch_size):
        lines = list(filter(("\n").__ne__, lines))

        self.lines = lines # list of strings
        collate = Collate()
        self.loader = torch.utils.data.DataLoader(self, batch_size=batch_size, num_workers=1, shuffle=True, collate_fn=collate)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].replace("\n", "")
        line = unidecode.unidecode(line) # remove special characters
        x = "".join(c for c in line if c not in "AEIOUaeiou") # remove vowels from input
        y = line
        return (x,y)

def encode_string(s):
    for c in s:
        if c not in string.printable:
            print(s)
    return [string.printable.index(c) + 1 for c in s]

def decode_labels(l):
    return "".join([string.printable[c - 1] for c in l])

class Collate:
    def __call__(self, batch):
        """
        batch: list of tuples (input string, output string)
        Returns a minibatch of strings, encoded as labels and padded to have the same length.
        """
        x = []; y = []
        batch_size = len(batch)
        for index in range(batch_size):
            x_,y_ = batch[index]
            x.append(encode_string(x_))
            y.append(encode_string(y_))

        # pad all sequences to have same length
        T = [len(x_) for x_ in x]
        U = [len(y_) for y_ in y]
        T_max = max(T)
        U_max = max(U)
        for index in range(batch_size):
            x[index] += [NULL_INDEX] * (T_max - len(x[index]))
            x[index] = torch.tensor(x[index])
            y[index] += [NULL_INDEX] * (U_max - len(y[index]))
            y[index] = torch.tensor(y[index])

        # stack into single tensor
        x = torch.stack(x)
        y = torch.stack(y)
        T = torch.tensor(T)
        U = torch.tensor(U)

        return (x,y,T,U)
    
class Trainer:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

    def train(self, dataset, print_interval = 20):
        train_loss = 0
        num_samples = 0
        self.model.train()
        pbar = tqdm(dataset.loader)
        for idx, batch in enumerate(pbar):
            x,y,T,U = batch
            x = x.to(self.model.device); y = y.to(self.model.device)
            batch_size = len(x)
            num_samples += batch_size
            loss = self.model.compute_loss(x,y,T,U)
            self.optimizer.zero_grad()
            pbar.set_description("%.2f" % loss.item())
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * batch_size
            if idx % print_interval == 0:
                self.model.eval()
                guesses = self.model.greedy_search(x,T)
                self.model.train()
                print("\n")
                for b in range(2):
                    print("input:", decode_labels(x[b,:T[b]]))
                    print("guess:", decode_labels(guesses[b]))
                    print("truth:", decode_labels(y[b,:U[b]]))
                    print("")
        train_loss /= num_samples
        return train_loss

    def test(self, dataset, print_interval=1):
    test_loss = 0
    num_samples = 0
    self.model.eval()
    pbar = tqdm(dataset.loader)
    for idx, batch in enumerate(pbar):
        x,y,T,U = batch
        x = x.to(self.model.device); y = y.to(self.model.device)
        batch_size = len(x)
        num_samples += batch_size
        loss = self.model.compute_loss(x,y,T,U)
        pbar.set_description("%.2f" % loss.item())
        test_loss += loss.item() * batch_size
        if idx % print_interval == 0:
            print("\n")
            print("input:", decode_labels(x[0,:T[0]]))
            print("guess:", decode_labels(self.model.greedy_search(x,T)[0]))
            print("truth:", decode_labels(y[0,:U[0]]))
            print("")
    test_loss /= num_samples
    return test_loss
    

