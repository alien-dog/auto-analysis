import torch
import torch.nn as nn

# Define the Siamese network
class SiameseNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # LSTM branch
        self.lstm_branch = nn.LSTM(input_size=384 * 2, hidden_size=384, num_layers=2, batch_first=True)

        # Distance prediction
        self.fc = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU()
        )
        self.out = nn.Linear(384, 1)

    def forward(self, x1, x2):
        # Concatenate the outputs from both branches
        combined = torch.cat((x1, x2), 1)
        # LSTM branch
        lstm_out, _ = self.lstm_branch(combined)
        # lstm_out = lstm_out[-1, :]  # Select the final LSTM output
        distance = self.fc(lstm_out) + lstm_out
        distance = self.out(distance)
        return distance


# Create the Siamese network
siamese_net = SiameseNetwork()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(siamese_net.parameters(), lr=0.001)

import numpy as np

# Define the shape of the input vectors and the number of samples
input_vector_shape = (384, 384)  # Example shape for image data
num_samples = 100  # The number of samples in your batch

# Generate random data for input_batch1
input_batch1 = np.random.randn(num_samples, *input_vector_shape).astype(np.float32)

# Make sure the data falls within a reasonable range (e.g., for image data, values between 0 and 255)
input_batch1 = (input_batch1 * 255).clip(0, 255)

# Convert the NumPy array to a PyTorch tensor
input_batch1 = torch.tensor(input_batch1, dtype=torch.float32)



# Generate random data for input_batch1
input_batch2 = np.random.randn(num_samples, *input_vector_shape).astype(np.float32)

# Make sure the data falls within a reasonable range (e.g., for image data, values between 0 and 255)
input_batch2 = (input_batch2 * 255).clip(0, 255)

# Convert the NumPy array to a PyTorch tensor
input_batch2 = torch.tensor(input_batch2, dtype=torch.float32)



# Generate random target distances for the sake of example
target_distances = np.random.rand(num_samples).astype(np.float32)

# If your target distances represent, for instance, Euclidean distances, you might want to scale them:
target_distances = target_distances * 100.0  # Scale for demonstration purposes

# Convert the NumPy array to a PyTorch tensor if needed
target_distances = torch.tensor(target_distances, dtype=torch.float32)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(len(input_batch1)):
        input1 = input_batch1[i]
        input2 = input_batch2[i]
        target_distance = target_distances[i]

        optimizer.zero_grad()
        output = siamese_net(input1, input2)
        loss = criterion(output, target_distance)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(input_batch1)}")

# Evaluate the model on a test set
# Create an instance of the model
# loaded_model = SiameseNetwork()
#
# # Load the model's state dictionary
# loaded_model.load_state_dict(torch.load('siamese_model.pth'))
#
# # Put the model in evaluation mode
# loaded_model.eval()
#
# # Now you can use the loaded model for predictions
# input1_new = torch.tensor(new_input_vector1, dtype=torch.float32)
# input2_new = torch.tensor(new_input_vector2, dtype=torch.float32)
#
# with torch.no_grad():
#     distance_prediction = loaded_model(input1_new, input2_new)

import torch

# Assuming you have a trained model (siamese_net)
# You can save the model's state dictionary like this
torch.save(siamese_net.state_dict(), 'siamese_model.pth')