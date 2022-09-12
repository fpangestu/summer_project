import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch

# Why F and G not in the classes?
classes =["0","1","2","3","4","5","6"]#,"f","g"] #["f","g"]
x = None
y = None

for c in classes:
    if x is None:
        x = np.load("/content/drive/MyDrive/summer/src/data/"+c+"/relcoords.npy")
        if c == "f":
            c = 0
        elif c == "g":
            c = 1
        y = np.ones(x.shape[0])*int(c)
    else:
        newx = np.load("/content/drive/MyDrive/summer/src/data/"+c+"/relcoords.npy")
        if c == " f":
            c = 0
        elif c == "g":
            c = 1
        x = np.concatenate((x, newx),axis=0)
        y = np.concatenate((y, np.ones(newx.shape[0])*int(c)), axis=0)

x = x.reshape(x.shape[0],-1)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2)

class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Linear(3 * 21, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 7))
            #torch.nn.Softmax())

        for c in self.block.children():
            try:
                torch.nn.init.xavier_normal_(c.weight)
            except:
                pass

    def forward(self, x):
        return self.block(x)


model = TinyModel()
learning_rate = 1e-4
batch_size = 16
epochs = 300

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_data = DataLoader(TensorDataset(torch.Tensor(x_train).type(torch.float32),torch.Tensor(y_train).type(torch.long)),
                        batch_size=batch_size, shuffle=True)
test_data = DataLoader(TensorDataset(torch.Tensor(x_test).type(torch.float32),torch.Tensor(y_test).type(torch.long)),
                       batch_size=batch_size, shuffle=True)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#epochs = 500
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_data, model, loss_fn, optimizer)
    test_loop(test_data, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "/content/drive/MyDrive/summer/src/torch_class/test_lab.pth")

'''
inp = Input(shape=x.shape[1])
x = Dense(64, activation="relu")(inp)
x = Dense(32, activation="relu")(x)
x = Dense(32, activation="relu")(x)
x = Dense(32, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x = Dense(7, activation="softmax")(x)
model = Model(inp, x)

model.compile(optimizer=Adam(learning_rate=1e-4),loss=SparseCategoricalCrossentropy(),metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=16,epochs=500,validation_data=(x_test,y_test))
model.save("./classifier")
'''