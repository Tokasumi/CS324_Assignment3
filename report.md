#### Task 1.1 Implement LSTM without `torch.nn.LSTM`

The forwarding function is shown here:

```python
    def forward(self, current_input, prev_hidden, prev_cell):
        
        concatenate = torch.cat((current_input, prev_hidden), 1)

        f = self.f_gate(concatenate)
        i = self.i_gate(concatenate)
        g = self.g_gate(concatenate)
        o = self.o_gate(concatenate)

        current_cell = g * i + prev_cell * f
        current_hidden = o * self.tanh(current_cell)
        current_output = self.hidden_to_out(current_hidden)

        return current_output, current_hidden, current_cell
```

`self.f_gate` is a linear layer. According to the LSTM formula, `g_gate` uses tanh activation while others uses sigmoid activation.

The figure below is the loss of LSTM and RNN network with default hyperparameters (x axis are cut to first 1000 records). 

![fig](./figs/lstm_test.png)

#### Task 1.2 Comparison between LSTM and RNN in Longer Sequence

**1. Training Configurations**

In this experiment, `Adam` optimizer will be used instead of `RMSProp`. 

Hyperparameters such as learning rates are specified for different models to reach their best performances.

**Result**

 Here are simple comparisons between LSTM and RNN with different sequence length.

![fig](./figs/seqlen5.png)

![fig](./figs/seqlen10.png)

![fig](./figs/seqlen15.png)

We can conclude that LSTM outperforms RNN in sequence prediction, but both RNN and LSTM failed to learn in length 20 sequence.

**2. Techniques (or Tricks)**

It's difficult to train the model on a long sequence with only one label and only one input that contains the information of label.

The trick is utilize the model trained on a short sequence as the pre-trained model, which can expand the ability of both RNN and LSTM to learn on a long sequence.

Psudocode:

```python
model = LSTM().init()
for length in [10, 20, 30, ...]:
    fit(model, data(length))
    eval(model, data(length))
```

**Result**

Both RNN and LSTM Performs perfectly (*Accuracy = 100%*) all the way to 150 long sequence before a memory shortage.

![fig](figs/seqlen_transfer150.png)

#### Task 2.1 GAN Model

To make the model training easier, the generator and discriminator are fully-connected.

```python
# G: latent -> 256 -> 512 -> 1024 -> 1 * 28 * 28
# D: 1 * 28 * 28 -> 1024 -> 512 -> 256 -> 1
```

Output of two models are processed by *sigmoid* and *BCELoss*.

Note that since the discriminator can always learn faster than the generator, in every step, the discriminator will be trained (perform back-propagation) according to the probability
$$
p_{\mathrm{trainD}}=\frac{e^{L(D)}}{e^{L(D)}+e^{L(G)}}
$$
and the generator will be trained in every step.

#### Task 2.2 GAN Model Results

