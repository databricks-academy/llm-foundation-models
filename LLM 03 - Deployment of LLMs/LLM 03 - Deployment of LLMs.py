# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Understanding and Applying Quantization
# MAGIC
# MAGIC Quantization is a method that can allow models to run faster and use less memory. By converting 32-bit floating-point numbers (the `float32` data type) into lower-precision formats, like 8-bit integers (the `int8` data type), we can reduce the computational requirements of our models. Let's start with the basics and gradually move towards quantizing complex models like CNNs.
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Explore how to quantize a a single variable and a function in pytorch
# MAGIC 1. Apply quantization to a neural network
# MAGIC 1. Compare the size and performance of quantized convolutional neural network 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import io

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 1 - Quantization
# MAGIC
# MAGIC We'll illustrate both 4-bit and 8-bit quantization. As for the neural network part, we'll create a simple model and show how to quantize and dequantize its weights. Since we can't download data or train models in this environment, I'll present the code you would use to do it.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Quantization of a Single Value
# MAGIC Quantization is the process of constraining an input from a large set to output in a smaller set. In the context of deep learning, it's used to reduce the precision of the weights and activations of the neural network models. This can help to reduce the memory footprint and computational intensity of models. Here, we'll start by quantizing a single floating point number.
# MAGIC
# MAGIC We'll define two functions: one to quantize a value and another to unquantize it. The quantize function will take a floating point number and a number of bits, and will output an integer representation of the input number. The unquantize function will take the integer and the number of bits, and will output the floating point number.
# MAGIC
# MAGIC The range of input values for the quantize function is between -1 and 1. The range of output values for the unquantize function is also between -1 and 1. The number of bits determines the precision of the quantization. More bits means higher precision, but more memory usage. For this demonstration, we'll use 4 and 8 bits.

# COMMAND ----------

# Let's start by defining the quantize and unquantize functions:

def quantize(value, bits):
    """
    Quantizes a floating point number to an integer, given a certain number of bits.
    The range is from -1.0 to 1.0.
    
    Args:
    value (float): The value to be quantized.
    bits (int): The number of bits used for quantization.
    
    Returns:
    int: The quantized value.
    """
    assert -1.0 <= value <= 1.0, "Value out of range"
    quantized_value = np.round(value * (2**(bits - 1) - 1))
    return int(quantized_value)

def unquantize(quantized_value, bits):
    """
    Unquantizes an integer back to a floating point number, given the original number of bits.
    The range is from -1.0 to 1.0.
    
    Args:
    quantized_value (int): The value to be unquantized.
    bits (int): The number of bits used for quantization.
    
    Returns:
    float: The unquantized value.
    """
    value = quantized_value / (2**(bits - 1) - 1)
    return float(value)

# COMMAND ----------

# Test the quantize and unquantize functions with 4 and 8 bits
value = 0.5
quantized_value_4bit = quantize(value, bits=4)
unquantized_value_4bit = unquantize(quantized_value_4bit, bits=4)

quantized_value_8bit = quantize(value, bits=8)
unquantized_value_8bit = unquantize(quantized_value_8bit, bits=8)

print(f"Original Value: {value}\n----\n4-bit Quantization:{quantized_value_4bit}\n4-bit Unquantization: {unquantized_value_4bit}\n----\n8-bit Quantization:{quantized_value_8bit}\n8-bit Unquantization: {unquantized_value_8bit}")

# COMMAND ----------

# MAGIC %md
# MAGIC The quantize and unquantize functions are working as expected. The float value 0.5 was quantized to 4 and 64 for 4-bit and 8-bit precision respectively. Then, the quantized values were unquantized back to approximately 0.5 (with some deviation due to the rounding operation in the quantization process).

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Quantization of a Function
# MAGIC Now let's demonstrate quantization and unquantization with a function. To simplify, we'll use the sine function as an example. For this demonstration, we'll generate values, quantize them, and then unquantize them. We'll plot the original, quantized, and unquantized values to visualize the effects of quantization.
# MAGIC
# MAGIC We'll start by generating the values of the sine function, then we'll quantize and unquantize those values. Finally, we'll plot the original, quantized, and unquantized values.

# COMMAND ----------

# # Generate values
x = np.linspace(-1, 1, 100)
y = np.sin(np.pi * x)

# Quantize and unquantize values for 4 and 8 bits
y_quantized_4bit = np.array([quantize(val, bits=4) for val in y])
y_unquantized_4bit = np.array([unquantize(val, bits=4) for val in y_quantized_4bit])

y_quantized_8bit = np.array([quantize(val, bits=8) for val in y])
y_unquantized_8bit = np.array([unquantize(val, bits=8) for val in y_quantized_8bit])

# Calculate quantization loss for 4 and 8 bits
loss_4bit = np.mean((y - y_unquantized_4bit)**2)
loss_8bit = np.mean((y - y_unquantized_8bit)**2)

print(f"Loss of 4-bit quantization: {loss_4bit}\nLoss of 8-bit quantization: {loss_8bit}")

# COMMAND ----------

# Plot original, quantized and unquantized values
plt.figure(figsize=(10, 12))

plt.subplot(4, 1, 1)
plt.plot(x, y, label="Original")
plt.title("Original")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.scatter(x, y_quantized_4bit, label="Quantized 4 bit", marker="s")
plt.legend()
plt.title("Quantized")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.scatter(x, y_quantized_8bit, label="Quantized 8 bit", marker="s")
plt.legend()
plt.title("Quantized")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(x, y_unquantized_4bit, label="Unquantized 4 bit")
plt.plot(x, y_unquantized_8bit, label="Unquantized 8 bit")
plt.legend()
plt.title("Unquantized")
plt.grid(True)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The plots illustrate the original sine function, the 4-bit and 8-bit quantized values, and their unquantized counterparts.
# MAGIC
# MAGIC The 'Quantized' plot clearly shows the "step" pattern of the quantized values. The 8-bit quantized values have more levels and are closer to the original function compared to the 4-bit values, which have fewer levels and deviate more.
# MAGIC
# MAGIC The 'Unquantized' plot shows the values obtained by converting the quantized values back to floating-point numbers. The 8-bit unquantized values are very close to the original function, while the 4-bit unquantized values deviate more due to the reduced precision.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3. Quantization of a Simple Neural Network
# MAGIC Next, let's apply quantization to a neural network. We'll create a simple network with one hidden layer, then we'll quantize and dequantize its weights.
# MAGIC
# MAGIC In PyTorch, [quantization](https://pytorch.org/docs/stable/quantization.html) is achieved using a `QuantStub` and `DeQuantStub` to mark the points in the model where the data needs to be converted to quantized form and converted back to floating point form, respectively. After defining the network with these stubs, we use the `torch.quantization.prepare` and `torch.quantization.convert` functions to quantize the model.
# MAGIC
# MAGIC The process of quantizing a model in PyTorch involves the following steps:
# MAGIC
# MAGIC - Define a neural network and mark the points in the model where the data needs to be converted to quantized form and converted back to floating point form. This is done using a `QuantStub` and `DeQuantStub`.
# MAGIC - Specify a quantization configuration for the model using `torch.quantization.get_default_qconfig`. This sets up the quantization parameters.
# MAGIC - Prepare the model for quantization using `torch.quantization.prepare`. This function replaces specified modules in the model with their quantized counterparts.
# MAGIC - Calibrate the model on a calibration dataset. During calibration, the model is run on a calibration dataset and the range of the activations is observed. This is used to determine the parameters for quantization.
# MAGIC - Convert the prepared and calibrated model to a quantized version using torch.quantization.convert. This function changes these modules to use quantized weights.

# COMMAND ----------

# Define the network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # QuantStub will act as a placeholder for the quantization process, it simulates quantization of inputs to the model.
        self.quant = torch.quantization.QuantStub()
        
        # Define two fully connected layers (aka linear layers) for our simple neural network
        self.fc1 = nn.Linear(28 * 28, 128)  # Input size is 28*28 (size of a flattened MNIST image), output size is 128
        self.fc2 = nn.Linear(128, 10)  # Input size is 128 (output of previous layer), output size is 10 (for 10 classes)

        # DeQuantStub simulates the dequantization of the final output of the model, converting it back to a floating point number.
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # Reshape the input tensor to a vector of size 28*28
        x = x.view(-1, 28 * 28)
        
        # Pass the input through the QuantStub, which will simulate the quantization of the input tensor
        x = self.quant(x)
        
        # Apply the first fully connected layer and ReLU activation function
        x = torch.relu(self.fc1(x))
        
        # Apply the second fully connected layer
        x = self.fc2(x)
        
        # Pass the output through the DeQuantStub, which will simulate the dequantization of the output tensor
        x = self.dequant(x)
        
        return x

# COMMAND ----------

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(root=DA.paths.working_dir, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Define loss function and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# COMMAND ----------

# Train the network
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print("[%d, %5d] loss: %.3f" %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print("Finished Training")

# COMMAND ----------

# Specify quantization configuration
net.qconfig = torch.ao.quantization.get_default_qconfig("onednn")

# Prepare the model for static quantization. This inserts observers in the model that will observe activation tensors during calibration.
net_prepared = torch.quantization.prepare(net)

# Now we convert the model to a quantized version.
net_quantized = torch.quantization.convert(net_prepared)

# Once the model is quantized, it can be used for inference in the same way as the unquantized model, but it will use less memory and potentially have faster inference times, at the cost of a possible decrease in accuracy.

# COMMAND ----------

# Let's look at the sizes of these two models on disk and see how much we save by quantization
buf = io.BytesIO()
torch.save(net.state_dict(), buf)
size_original = sys.getsizeof(buf.getvalue())

buf = io.BytesIO()
torch.save(net_quantized.state_dict(), buf)
size_quantized = sys.getsizeof(buf.getvalue())

print("Size of the original model: ", size_original)
print("Size of the quantized model: ", size_quantized)
print(f"The quantized model is {np.round(100.*(size_quantized )/ size_original)}% the size of the original model")

# COMMAND ----------

# Print out the weights of the original network
for name, param in net.named_parameters():
    print("Original Network Layer:", name)
    print(param.data)

# COMMAND ----------

# Print out the weights of the quantized network
for name, module in net_quantized.named_modules():
    if isinstance(module, nn.quantized.Linear):
        print("Quantized Network Layer:", name)
        
        print("Weight:")
        print(module.weight())
        
        print("Bias:")
        print(module.bias)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Comparing a Quantized and Non-Quantized Model
# MAGIC
# MAGIC Here is a summary of the details and a comparison with the original model:
# MAGIC
# MAGIC - `Tensor Values`: In the quantized model, these are quantized values of the weights and biases, compared to the original model which stores these in floating point precision. These values are used in the computations performed by the layer, and they directly affect the layer's output.
# MAGIC - `Size`: This is the shape of the weight or bias tensor and it should be the same in both the original and quantized model. In a fully-connected layer, this corresponds to the number of neurons in the current layer and the number of neurons in the previous layer.
# MAGIC - `Dtype`: In the original model, the data type of the tensor values is usually torch.float32 (32-bit floating point), whereas in the quantized model it is a quantized data type like torch.qint8 (8-bit quantized integer). This reduces the memory usage and computational requirements of the model.
# MAGIC - `Quantization_scheme`: This is specific to the quantized model. It is the type of quantization used, for example, torch.per_channel_affine means different channels (e.g., neurons in a layer) can have different scale and zero_point values.
# MAGIC - `Scale & Zero Point`: These are parameters of the quantization process and are specific to the quantized model. They are used to convert between the quantized and dequantized forms of the tensor values.
# MAGIC - `Axis`: This indicates the dimension along which the quantization parameters vary. This is also specific to the quantized model.
# MAGIC - `Requires_grad`: This indicates whether the tensor is a model parameter that is updated during training. It should be the same in both the original and quantized models.

# COMMAND ----------

# Suppose we have some input data
input_data = torch.randn(1, 28 * 28)

# We can pass this data through both the original and quantized models
output_original = net(input_data)
output_quantized = net_quantized(input_data)

# The outputs should be similar, because the quantized model is a lower-precision
# approximation of the original model. However, they won't be exactly the same
# because of the quantization process.
print("Output from original model:", output_original.data)
print("Output from quantized model:", output_quantized.data)

# COMMAND ----------

# The difference between the outputs is an indication of the "quantization error",
# which is the error introduced by the quantization process.
quantization_error = (output_original - output_quantized).abs().mean()
print("Quantization error:", quantization_error)

# COMMAND ----------

# The weights of the original model are stored in floating point precision, so they
# take up more memory than the quantized weights. We can check this using the
# `element_size` method, which returns the size in bytes of one element of the tensor.
print(f"Size of one weight in original model: {net.fc1.weight.element_size()} bytes (32bit)")
print(f"Size of one weight in quantized model: {net_quantized.fc1.weight().element_size()} byte (8bit)")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC This example shows how the quantized model can be used in the same way as the original model. It also demonstrates the trade-off between precision and memory usage/computation speed that comes with quantization. The quantized model uses less memory and is faster to compute, but the outputs are not exactly the same as the original model due to the quantization error.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
