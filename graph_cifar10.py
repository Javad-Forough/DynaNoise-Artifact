import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['legend.title_fontsize'] = 'x-large'
mpl.rcParams['legend.fontsize'] = 12

def plot_metrics_side_by_side(
    csv_file,
    x_col,
    x_label,
    out_png
):
    """
    Creates a single row of 4 subplots (side by side):
      1) test_acc: plots two lines (test_acc_no_def, test_acc_dyna)
      2) conf_acc: plots one line (conf_acc_dyna)
      3) loss_acc: plots one line (loss_acc_dyna)
      4) shadow_acc: plots one line (shadow_acc_dyna)
    """

    # Read CSV and ensure x_col is float
    df = pd.read_csv(csv_file)
    df[x_col] = df[x_col].astype(float)
    x_vals = df[x_col]

    # Create a figure with 4 horizontal subplots
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(18, 4), sharex=False)

    #####################################
    # Subplot 1: test_acc
    #####################################
    ax = axs[0]

    # We'll assume columns: test_acc_no_def, test_acc_dyna exist
    if "test_acc_no_def" in df.columns and "test_acc_dyna" in df.columns:
        y_no_def = df["test_acc_no_def"]
        y_dyna   = df["test_acc_dyna"]

        ax.plot(x_vals, y_no_def, color="darkorange",  label="No Defense", linewidth=4)
        ax.plot(x_vals, y_dyna, color="blue", linestyle="--", label="DynaNoise", linewidth=4)

        # auto-scale y-limits
        y_min = min(y_no_def.min(), y_dyna.min())
        y_max = max(y_no_def.max(), y_dyna.max())
        margin = 0.005
        y_lower = max(0, y_min - margin)
        y_upper = min(1, y_max + margin)
        ax.set_ylim(y_lower, y_upper)

        ax.set_title("Test Accuracy", fontweight='bold')
        ax.legend(loc="best")
        ax.grid(True)
    else:
        ax.set_title("test_acc columns missing", fontweight='bold')

    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel("Accuracy", fontweight='bold')


    #####################################
    # Subplot 2: conf_acc (only Dyna)
    #####################################
    ax = axs[1]
    if "conf_acc_dyna" in df.columns:
        y_conf_dyna = df["conf_acc_dyna"]
        ax.plot(x_vals, y_conf_dyna, color="red", label="Confidence (Dyna)", linewidth=4)
        
        # auto-scale y-limits
        y_min = y_conf_dyna.min()
        y_max = y_conf_dyna.max()
        margin = 0.005
        y_lower = max(0, y_min - margin)
        y_upper = min(1, y_max + margin)
        ax.set_ylim(y_lower, y_upper)

        ax.set_title("Confidence Attack", fontweight='bold')
        ax.legend(loc="best")
        ax.grid(True)
    else:
        ax.set_title("conf_acc_dyna missing", fontweight='bold')

    ax.set_xlabel(x_label, fontweight='bold')


    #####################################
    # Subplot 3: loss_acc (only Dyna)
    #####################################
    ax = axs[2]
    if "loss_acc_dyna" in df.columns:
        y_loss_dyna = df["loss_acc_dyna"]
        ax.plot(x_vals, y_loss_dyna, color="green", label="Loss (Dyna)", linewidth=4)

        y_min = y_loss_dyna.min()
        y_max = y_loss_dyna.max()
        margin = 0.005
        y_lower = max(0, y_min - margin)
        y_upper = min(1, y_max + margin)
        ax.set_ylim(y_lower, y_upper)

        ax.set_title("Loss Attack", fontweight='bold')
        ax.legend(loc="best")
        ax.grid(True)
    else:
        ax.set_title("loss_acc_dyna missing", fontweight='bold')

    ax.set_xlabel(x_label, fontweight='bold')


    #####################################
    # Subplot 4: shadow_acc (only Dyna)
    #####################################
    ax = axs[3]
    if "shadow_acc_dyna" in df.columns:
        y_shadow_dyna = df["shadow_acc_dyna"]
        ax.plot(x_vals, y_shadow_dyna, color="purple", label="Shadow (Dyna)", linewidth=4)

        y_min = y_shadow_dyna.min()
        y_max = y_shadow_dyna.max()
        margin = 0.005
        y_lower = max(0, y_min - margin)
        y_upper = min(1, y_max + margin)
        ax.set_ylim(y_lower, y_upper)

        ax.set_title("Shadow Attack", fontweight='bold')
        ax.legend(loc="best")
        ax.grid(True)
    else:
        ax.set_title("shadow_acc_dyna missing", fontweight='bold')

    ax.set_xlabel(x_label, fontweight='bold')


    # plt.suptitle(title, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle

    # Ensure out directory
    out_dir = os.path.dirname(out_png)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"[INFO] Plot saved to {out_png}")


if __name__ == "__main__":
    # Example usage:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    fig_folder = os.path.join(base_dir, "fig")

    # We'll assume e.g. "results_cifar10_bv.csv" with columns:
    # base_variance, test_acc_no_def, test_acc_dyna,
    # conf_acc_dyna, loss_acc_dyna, shadow_acc_dyna
    
    
    # csv_file = "results_cifar10_bv.csv"
    # x_col = "base_variance"
    # x_label = "Base Variance"
    # title = "CIFAR-10 (AlexNet): No Defense vs. Dyna (Test Acc) and Dyna Only (Attacks)"

    # csv_file = "results_cifar10_ls.csv"
    # x_col = "lambda_scale"
    # x_label = "Lambda Scale"
    # title = "CIFAR-10 (AlexNet): No Defense vs. Dyna (Test Acc) and Dyna Only (Attacks)"


    csv_file = "results_cifar10_t.csv"
    x_col = "temperature"
    x_label = "Temperature"



    # Construct an output path
    # out_png = os.path.join(fig_folder, "cifar10_base_variance_comparison.png")
    # out_png = os.path.join(fig_folder, "cifar10_lambda_scale_comparison.png")
    out_png = os.path.join(fig_folder, "cifar10_temperature_comparison.png")


    plot_metrics_side_by_side(csv_file, x_col, x_label, out_png)
