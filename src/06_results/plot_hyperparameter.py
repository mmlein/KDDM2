import pandas as pd
import matplotlib.pyplot as plt
import string


def plot_results(feature, input_filename, output_filename):
    optimization_data = pd.read_csv(input_filename)
    result = optimization_data.groupby([feature]).mean().reset_index()
    alphabet = list(string.ascii_uppercase)
    if "sum" in optimization_data.columns:
        alphabet.append("sum")
    fig = plt.figure(figsize=(20, 10))

    for letter in alphabet:
        if letter == "sum":
            plt.plot(result[feature], result[letter],
                     "b-", label=letter, linewidth=3)
        else:
            plt.plot(result[feature], result[letter],
                     ":", label=letter, linewidth=1)
        plt.title(letter)

    plt.xlabel(feature)
    plt.ylabel("Percentage of outliers")
    plt.title(f"Effect of {feature}")
    plt.legend()
    plt.grid(color='#DDDDDD', linestyle=':', linewidth=0.5)
    plt.savefig(output_filename)
    plt.show()


# plot_results("%", "Effect of the %", "Critical Values", "local_outlier_probability_hyper_parameter_tuning_with_outlier_3std_06_07_n10.csv", "local_outlier_probability_hyper_parameter_tuning_with_outlier_2std_0.6.pdf")

# print()