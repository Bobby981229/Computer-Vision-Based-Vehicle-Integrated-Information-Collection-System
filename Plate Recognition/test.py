import matplotlib.pyplot as plt

epochs = 20
acc_array = [1218, 519, 493, 396, 382, 341, 347, 339, 328,
             325, 325, 325, 315, 320, 307, 339, 318, 306, 306, 298]


plt.plot(range(0, epochs, 1), acc_array, color='r', label='loss')  # Generate the plot
plt.xlabel('epochs'), plt.ylabel('Loss'), plt.title("Epochs - Loss Plot")
plt.legend(), plt.show()  # Display plot