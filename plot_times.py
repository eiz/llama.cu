import matplotlib.pyplot as plt


def read_data_from_file(filename):
    with open(filename, "r") as file:
        offset = int(file.readline().strip())
        floats = [float(line.strip()) for line in file]
    return offset, floats


# Read data and offset from the text file
filename = "times.txt"
offset, data = read_data_from_file(filename)

# Generate X-axis values with the given offset
x_values = range(offset, offset + len(data))

# Create the plot
plt.plot(x_values, data)
plt.xlabel("Steps")
plt.ylabel("Time (seconds)")
plt.title("Time per Token")


# Save the plot as 'timings.png' with 200dpi resolution
plt.savefig("timings.png", dpi=200)
