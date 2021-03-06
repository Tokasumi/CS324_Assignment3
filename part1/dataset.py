import math
import sys

import numpy as np
import torch.utils.data as data


class PalindromeDataset(data.Dataset):

    def __init__(self, seq_length):
        self.seq_length = seq_length

    def __len__(self):
        # Number of possible palindroms can be very big:
        # (10**(seq_length/2) or (10**((seq_length+1)/2)
        # Therefore we return the maximum integer value
        return sys.maxsize

    def __getitem__(self, idx):
        # Keep last digit as target label. Note: one-hot encoding for inputs is
        # more suitable for training, but this also works.
        full_palindrome = self.generate_palindrome()
        label = np.zeros(10, dtype=np.float32)
        label[int(full_palindrome[-1])] = 1.0

        # Split palindrome into inputs (N-1 digits) and target (1 digit)
        return full_palindrome[0:-1], label

    def generate_palindrome(self):
        # Generates a single, random palindrome number of 'length' digits.
        left = [np.random.randint(0, 10) for _ in range(math.ceil(self.seq_length / 2))]
        left = np.asarray(left, dtype=np.float32)
        right = np.flip(left, 0) if self.seq_length % 2 == 0 else np.flip(left[:-1], 0)
        return np.concatenate((left, right))


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = PalindromeDataset(6 + 1)
    data_loader = DataLoader(dataset, batch_size=4, num_workers=1)
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        print('=' * 50)
        print(batch_inputs.shape, batch_targets.shape)
        print(batch_inputs)
        print(batch_targets)
        if step >= 10:
            break
