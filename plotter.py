import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nilmtk.losses as losses
from skimage.metrics import structural_similarity
from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import uniform_filter1d
import nilmtk.utils as utils
import matplotlib.dates as mdates

RANGES = {
    "ukdale": {
        "fridge": (250, 2000),
        # "kettle": (7500, 9500),
        "kettle": (8500, 9100),
        # "microwave": (68400, 69500),
        "microwave": (68450, 68850),
        # "dish washer": (83500, 85100),
        "dish washer": (55000, 60000),
        "washing machine": (121800, 122500)
    }
}

GRAPH_NAMES = {
    "SGN": "SGN",
    "AttentionCNN": "Attention CNN",
    "DM": "U-Net-DM",
    "DMGated": "GT-NILM$^\mathrm{GM}$"
}

GRAPH_NAMES_FT = {
    "SGN": "SGN",
    "DMGated": "GT-NILM$^\mathrm{7-day}$"
}

APP_NAMES = {
    "kettle": "Kettle",
    "fridge": "Refrigerator",
    "microwave": "Microwave oven",
    "dish washer": "Dish washer",
    "washing machine": "Washing machine"
}

REDD_DF_MAP = {
    'fridge': {
        "AttentionCNN": "redd-dish washer+washing machine+fridge-AttentionCNN.csv",
        "SGN": "redd-dish washer+washing machine+fridge-SGN.csv",
        # "EnerGAN": "ukdale-fridge-ENER_GAN.csv",
        "DM": "../-redd-dish washer+fridge+microwave+washing machine-DM_GATE2.csv",
        "DMGated": "gated-redd-dish washer+fridge+microwave+washing machine-DM_GATE2.csv"
    },
    'washing machine': {
        "AttentionCNN": "redd-dish washer+washing machine+fridge-AttentionCNN.csv",
        "SGN": "redd-dish washer+washing machine+fridge-SGN.csv",
        # "EnerGAN": "ukdale-washing machine-ENER_GAN.csv",
        "DM": "../-redd-dish washer+fridge+microwave+washing machine-DM_GATE2.csv",
        "DMGated": "gated-redd-dish washer+fridge+microwave+washing machine-DM_GATE2.csv"
    },
    'dish washer': {
        "AttentionCNN": "redd-dish washer+washing machine+fridge-AttentionCNN.csv",
        "SGN": "redd-dish washer+washing machine+fridge-SGN.csv",
        # "EnerGAN": "ukdale-dish washer-ENER_GAN.csv",
        "DM": "../-redd-dish washer+fridge+microwave+washing machine-DM_GATE2.csv",
        "DMGated": "gated-redd-dish washer+fridge+microwave+washing machine-DM_GATE2.csv"
    },
    'microwave': {
        "AttentionCNN": "redd-microwave-AttentionCNN.csv",
        "SGN": "redd-microwave-SGN.csv",
        # "EnerGAN": "ukdale-microwave-ENER_GAN.csv",
        "DM": "../-redd-dish washer+fridge+microwave+washing machine-DM_GATE2.csv",
        "DMGated": "gated-redd-dish washer+fridge+microwave+washing machine-DM_GATE2.csv"
    }
}

UKDALE_DF_MAP = {
    'fridge': {
        "AttentionCNN": "ukdale-fridge-AttentionCNN.csv",
        "SGN": "ukdale-fridge-SGN.csv",
        "DM": "ukdale-kettle+fridge-DM_GATE2.csv",
        "DMGated": "gated-ukdale-dish washer+fridge+kettle+microwave+washing machine-DM_GATE2.csv"
    },
    'kettle': {
        "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
        "SGN": "ukdale-kettle-SGN.csv",
        "DM": "ukdale-kettle+fridge-DM_GATE2.csv",
        "DMGated": "gated-ukdale-dish washer+fridge+kettle+microwave+washing machine-DM_GATE2.csv",
    },
    'microwave': {
        "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
        "SGN": "ukdale-microwave-SGN.csv",
        "DM": "ukdale-dish washer+washing machine+microwave-DM_GATE2.csv",
        "DMGated": "gated-ukdale-dish washer+fridge+kettle+microwave+washing machine-DM_GATE2.csv",
    },
    'washing machine': {
        "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
        "SGN": "ukdale-washing machine-SGN.csv",
        "DM": "../-ukdale-dish washer+fridge+kettle+microwave+washing machine-DM_GATE2.csv",
        "DMGated": "gated-ukdale-dish washer+fridge+kettle+microwave+washing machine-DM_GATE2.csv"
    },
    'dish washer': {
        "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
        "SGN": "ukdale-dish washer-SGN.csv",
        "DM": "../-ukdale-dish washer+fridge+kettle+microwave+washing machine-DM_GATE2.csv",
        "DMGated": "gated-ukdale-dish washer+fridge+kettle+microwave+washing machine-DM_GATE2.csv"
    }
}

# CSFONT = {'fontname': 'Times New Roman'}


LINE_STYLES = {
    'AttentionCNN': 'dashdot',
    'SGN': 'dashed',
    'DM': 'dotted',
    'DMGated': 'solid'
}

LINE_COLORS = {
    'AttentionCNN': '#ff7f0e',
    'SGN': '#2ca02c',
    'DM': '#d62728',
    'DMGated': '#9467bd'
}


def load_app_gate(app_name, path):
    truth_name = app_name + "_truth"
    pred_name = app_name + "_pred"
    gate_name = app_name + "_gate"
    ungated_name = app_name + "_ungated"

    df_orig = pd.read_csv(path, index_col=0)

    df = df_orig.loc[:, ("mains", truth_name, pred_name, gate_name, ungated_name)]
    df.columns = ["mains", "truth", "pred", "gate", "ungated"]

    return df


def plot_app_gate(app, path: str, rng):
    df = load_app_gate(app, path)
    # rng = RANGES["ukda"][app]
    plt.figure(figsize=(12, 8))

    # plt.subplots(2, 1, figsize=(12, 6))
    plt.title(APP_NAMES[app], fontweight='bold')

    mains = df.loc[:, "mains"].to_numpy()
    truth = df.loc[:, "truth"].to_numpy()
    pred = df.loc[:, "pred"].to_numpy()
    gate = df.loc[:, "gate"].to_numpy() * 2000
    ung = df.loc[:, "ungated"].to_numpy()

    x_val = np.arange(0, rng[1] - rng[0], 1)

    plt.subplot(2, 1, 1)
    plt.plot(x_val, mains[rng[0]: rng[1]], linewidth=1, label="Mains")
    plt.fill_between(x_val, 0, truth[rng[0]:rng[1]],
                     label="Ground truth", color="#aaaaaa77")
    plt.plot(x_val, pred[rng[0]: rng[1]], linewidth=1, label="Predicted")

    plt.subplot(2, 1, 2)
    plt.fill_between(x_val, 0, truth[rng[0]:rng[1]],
                     label="Ground truth", color="#aaaaaa77")
    # plt.plot(x_val, gate[rng[0]: rng[1]], linewidth=1, label="Gate signal")
    plt.plot(x_val, pred[rng[0]: rng[1]], linewidth=2, label="Predicted")
    plt.plot(x_val, ung[rng[0]: rng[1]], linewidth=1, label="Original")

    plt.xlabel("Samples")
    plt.ylabel("Power (W)")
    plt.legend(loc="upper right")

    save_path = path[path.rfind("/")].replace('csv', 'png')
    plt.savefig(f"./csv_ft/figures/{save_path}", bbox_inches='tight')

    plt.show()


def plot_app_transfer_multi(app, clf_paths, rng):
    df = load_app(app, clf_paths)
    print(df.head())

    plt.figure(figsize=(12, 6))

    print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    plt.title(APP_NAMES[app], fontweight='bold')

    mains = df.loc[:, "mains"].to_numpy()
    truth = df.loc[:, "truth"].to_numpy()

    clf_data = {clf: df.loc[:, clf].to_numpy() for clf in clf_paths}

    x_val = np.arange(0, rng[1] - rng[0], 1)

    plt.xlim((0, rng[1] - rng[0]))

    plt.plot(x_val, mains[rng[0]: rng[1]], linewidth=1, label="Aggregated power")
    plt.fill_between(x_val, 0, truth[rng[0]:rng[1]],
                     label="Ground truth", color="#aaaaaa77")
    for clf, data in clf_data.items():
        plt.plot(x_val, data[rng[0]: rng[1]], linewidth=2, label=GRAPH_NAMES_FT[clf],
                 color=LINE_COLORS[clf], linestyle=LINE_STYLES[clf])

    plt.xlabel("Time interval index")
    plt.ylabel("Power (W)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    save_path = f"./csv_ft/figures/tsf_{app}.svg"
    plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_app_transfer(app, path: str, rng):
    df = load_app_gate(app, path)
    # rng = RANGES["ukda"][app]
    plt.figure(figsize=(12, 6))

    print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    plt.title(APP_NAMES[app], fontweight='bold')

    mains = df.loc[:, "mains"].to_numpy()
    truth = df.loc[:, "truth"].to_numpy()
    pred = df.loc[:, "pred"].to_numpy()

    x_val = np.arange(0, rng[1] - rng[0], 1)

    plt.xlim((0, rng[1] - rng[0]))

    # plt.figure(figsize=(12, 6))
    plt.plot(x_val, mains[rng[0]: rng[1]], linewidth=1, label="Aggregated power")
    plt.fill_between(x_val, 0, truth[rng[0]:rng[1]],
                     label="Ground truth", color="#aaaaaa77")
    plt.plot(x_val, pred[rng[0]: rng[1]], linewidth=2, label="GT-NILM",
             color='#9467bd')

    plt.xlabel("Time interval index")
    plt.ylabel("Power (W)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    save_path = path[path.rfind("/") + 1:].replace('csv', 'svg')
    plt.savefig(f"./csv_ft/figures/tsf_{save_path}", bbox_inches='tight')

    plt.show()


def load_app_ds(app_name, ds: str):
    if ds.lower() == "redd":
        clf_dict = REDD_DF_MAP[app_name]
    elif ds.lower() == "ukdale":
        clf_dict = UKDALE_DF_MAP[app_name]
    else:
        raise ValueError

    return load_app(app_name, clf_dict)


def load_app(app_name, clf_dict: dict):
    truth_name = app_name + "_truth"
    pred_name = app_name + "_pred"

    ix = None
    df_list = []
    for clf, csv_file in clf_dict.items():
        subframe = pd.read_csv("results/" + csv_file, index_col=0)
        if ix is None:
            ix = subframe.index
        else:
            ix = ix.intersection(subframe.index)
        df_list.append((clf, subframe))

    df = pd.DataFrame()
    for clf, subframe in df_list:
        subframe = subframe.loc[ix]

        if len(df) == 0:
            df = subframe.loc[:, ("mains", truth_name, pred_name)]
            df.columns = ["mains", "truth", clf]
            df.index = ix
        else:
            df[clf] = subframe[pred_name]

    return df


# def compute_scores(clf, ):
#     pass


def find_best_window(df, clf, window_size, app_meta, metric="mse", step=200, sample_rate=6,
                     lambda_factor=0.5):
    """
    Find the best window in a DataFrame that:
    1. Contains at least one "truth > threshold" case.
    2. Has the minimum difference between truth and pred (using MSE, MAE, etc.).
    3. Uses 'step' to control the stride for efficiency.

    Returns:
    - (start_index, end_index): Tuple indicating the best window range, or None if no valid window is found.
    """

    # Convert to NumPy arrays for faster operations
    truth = df["truth"].to_numpy()
    pred = df[clf].to_numpy()

    best_start = None
    best_score = -float("inf")

    threshold = app_meta['on']
    min_on = app_meta['min_on'] / sample_rate

    # Compute min/max errors for normalization
    all_errors = []

    for start in range(0, len(df) - window_size + 1, step):
        end = start + window_size
        truth_window = truth[start:end]
        pred_window = pred[start:end]

        # Compute activation count
        activation_count = np.sum(truth_window > threshold)

        # Compute error
        if metric == "mse":
            error = np.mean((truth_window - pred_window) ** 2)
        elif metric == "mae":
            error = np.mean(np.abs(truth_window - pred_window))
        else:
            raise ValueError("Unsupported metric. Use 'mse' or 'mae'.")

        all_errors.append(error)

    # Normalize error values
    error_min = min(all_errors)
    error_max = max(all_errors)

    # Ensure we don't divide by zero
    error_range = error_max - error_min if error_max > error_min else 1

    for start in range(0, len(df) - window_size + 1, step):
        end = start + window_size
        truth_window = truth[start:end]
        pred_window = pred[start:end]

        # Compute activation count
        activation_count = np.sum(truth_window > threshold)

        # Compute error
        if metric == "mse":
            error = np.mean((truth_window - pred_window) ** 2)
        else:  # MAE
            error = np.mean(np.abs(truth_window - pred_window))

        # Normalize error to [0,1]
        norm_error = (error - error_min) / error_range

        # Compute score (higher is better)
        activation_ratio = activation_count / window_size
        score = lambda_factor * activation_ratio - (1 - lambda_factor) * norm_error

        # Update best window if score is higher
        if score > best_score and activation_count >= min_on:
            best_score = score
            best_start = start

    # Return best window indices
    if best_start is not None:
        return best_start, best_start + window_size
    else:
        return None  # No valid window found


def plot_app(app, df, clf_names, ds, score=True, cut=None, rng=None, note='', filter_invalid=False):
    if rng is None:
        rng = RANGES[ds][app]
    plt.figure(figsize=(12, 6))
    plt.title(APP_NAMES[app], fontweight='bold')

    filtering = filter_invalid_data if filter_invalid else lambda d, o, m: d

    if cut is not None:
        plt.ylim((0, cut))

    mains = df.loc[:, "mains"].to_numpy()
    truth = df.loc[:, "truth"].to_numpy()

    x_val = np.arange(0, rng[1] - rng[0], 1)

    plt.xlim((0, rng[1] - rng[0]))

    plt.plot(x_val, mains[rng[0]: rng[1]], linewidth=2, label="Aggregated power")
    plt.fill_between(x_val, 0, truth[rng[0]:rng[1]],
                     label="Ground truth", color="#aaaaaa77")

    meta = utils.APP_META[ds][app]
    if score:
        valid_truth = filtering(truth, meta['on'], meta['max'])
    else:
        valid_truth = truth

    line_style_index = 0
    for clf in clf_names:
        pred = df.loc[:, clf].to_numpy()
        # x_val = np.arange(rng[1] - rng[0]) * 6

        if score:
            valid_pred = filtering(pred, meta['on'], meta['max'])

            acc = losses.accuracy(app, truth, pred)
            f1 = losses.f1score(app, truth, pred)
            pre = losses.precision(app, truth, pred)
            recall = losses.recall(app, truth, pred)
            mae = losses.mae(app, truth, pred)
            sae = losses.sae(app, truth, pred)
            wssim = losses.wssim(app, valid_truth, valid_pred, window_size=11)

            print(f"{clf}: {acc:.4f}, F1: {f1:.3f}, MAE: {mae:.2f}, SAE: {sae:.2f}, "
                  f"WSSIM:{wssim:.3f}, Precision: {pre:.3f}, Recall: {recall:.3f}, ")

        plt.plot(x_val, pred[rng[0]: rng[1]], linewidth=2,
                 label=GRAPH_NAMES[clf], linestyle=LINE_STYLES[clf], color=LINE_COLORS[clf])
        line_style_index += 1

    plt.xlabel("Time interval index")
    plt.ylabel("Power (W)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(f"./figures/{ds}_{app}{'_' if note else ''}{note}.svg", bbox_inches='tight')

    plt.show()


def plot_app_real_time(app, dfs, clf_names, ds, rngs, meta=None, note='', mul=1.0):
    n_rows = len(rngs)
    # plt.figure(figsize=(12, 5 * n_rows + 1))
    # plt.subplots(n_rows, 1)

    plt.title(APP_NAMES[app], fontweight='bold')

    if meta is not None:
        plt.ylim((0, meta['max']))

    for i in range(len(rngs)):
        # plt.subplot(n_rows, 1, i + 1)
        rng = rngs[i]
        df = dfs[i]

        fig, ax = plt.subplots(figsize=(12, 5))

        # x_val = np.arange(0, rng[1] - rng[0], 1)

        mains = df.loc[:, "mains"].to_numpy()
        truth = df.loc[:, "truth"].to_numpy()

        if meta is not None:
            truth = filter_invalid_data(truth, meta['on'], meta['max'])

        times = df.index[:]
        x_val = pd.to_datetime(times[rng[0]: rng[1]]).tz_convert(None)

        # ax.xlim((0, rng[1] - rng[0]))

        ax.plot(x_val, mains[rng[0]: rng[1]], linewidth=2, label="Aggregated power")
        ax.fill_between(x_val, 0, truth[rng[0]:rng[1]],
                        label="Ground truth", color="#aaaaaa77")

        # Format x-axis to show Hour:Minute
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format as HH:MM
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=9))

        line_style_index = 0
        for clf in clf_names:
            pred = df.loc[:, clf].to_numpy() * mul
            # x_val = np.arange(rng[1] - rng[0]) * 6

            plt.plot(x_val, pred[rng[0]: rng[1]], linewidth=2,
                     label=GRAPH_NAMES[clf], linestyle=LINE_STYLES[clf])
            line_style_index += 1

        plt.ylabel("Power (W)")
        if i == 0:
            plt.legend(loc="upper left")

        plt.xlabel("Time")
        plt.tight_layout()

        plt.savefig(f"./figures/{ds}_{app}{'_' if note else ''}{note}-{i}.svg", bbox_inches='tight')

        plt.show()


def draw_ukdale_real_time():
    # df_wash = load_app("washing machine", {
    #     "DMGated": "gated-ukdale-washing machine-DM_GATE2.csv"
    # })
    #
    # print(df_wash.head())
    # plot_app("washing machine",
    #          df_wash,
    #          ["DMGated"],
    #          'ukdale', score=False,
    #          rng=(121320, 122040),
    #          note="realtime")

    df_dish1 = load_app("dish washer", {
        "DMGated": "../gated-special1-ukdale-dish washer-DM_GATE2.csv"
    })
    # df_dish2 = load_app("dish washer", {
    #     "DMGated": "../gated-special-ukdale-dish washer-DM_GATE2.csv"
    # })
    df_dish2 = load_app("dish washer", {
        "DMGated": "gated-ukdale-dish washer+fridge+kettle+microwave+washing machine-DM_GATE2.csv"
        # "DMGated": "gated-ukdale-dish washer-DM_GATE2.csv"
    })

    rng = find_best_window(df_dish2, "DMGated", 720,
                           utils.GENERAL_APP_META['dish washer'], "mse")

    plot_app_real_time("dish washer",
                       [df_dish1, df_dish2],
                       ["DMGated"],
                       'ukdale',
                       meta=utils.GENERAL_APP_META['dish washer'],
                       rngs=[(rng[0] - 460, rng[1] - 460), (rng[0] + 50, rng[1] + 50)],
                       note="realtime",
                       mul=1.2)


def draw_direct():
    # df_fridge = load_app("fridge", {
    #     "AttentionCNN": "ukdale-fridge-AttentionCNN.csv",
    #     "SGN": "ukdale-fridge-SGN.csv",
    #     "DM": "ukdale-kettle+fridge-DM_GATE2.csv",
    #     "DMGated": "gated-ukdale-fridge-DM_GATE2.csv"
    # })
    #
    # best_window_fr = RANGES['ukdale']['fridge']
    # # best_window_fr = find_best_window(df_fridge, "DMGated", 1800,
    # #                                   utils.GENERAL_APP_META['fridge'],
    # #                                   step=100, sample_rate=6, metric='mae')
    # #
    # # print("fr best window", best_window_fr)
    #
    # plot_app("fridge",
    #          df_fridge,
    #          [
    #              "AttentionCNN",
    #              "SGN",
    #              "DM",
    #              "DMGated"],
    #          'ukdale', score=True,
    #          rng=(best_window_fr[0], best_window_fr[1]),
    #          cut=600)

    df_dish = load_app("dish washer", {
        "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
        "SGN": "ukdale-dish washer-SGN.csv",
        # "DM": "ukdale-dish washer+washing machine+microwave-DM_GATE2.csv",
        "DM": UKDALE_DF_MAP['dish washer']['DM'],
        # "DMGated": "gated-ukdale-dish washer-DM_GATE2.csv"
        "DMGated": UKDALE_DF_MAP['dish washer']['DMGated']
    })

    best_window_dw = find_best_window(df_dish, "DMGated", 1200,
                                      utils.GENERAL_APP_META['dish washer'],
                                      step=200, sample_rate=6, metric='mae')

    print("dish best window", best_window_dw)

    print(df_dish.head())
    plot_app("dish washer",
             df_dish,
             [
                 "AttentionCNN",
                 "SGN",
                 "DM",
                 "DMGated"],
             'ukdale', score=True,
             rng=best_window_dw)

    # df_wash = load_app("washing machine", {
    #     "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
    #     "SGN": "ukdale-washing machine-SGN.csv",
    #     # "DM": "ukdale-dish washer+washing machine+microwave-DM_GATE2.csv",
    #     "DM": UKDALE_DF_MAP['washing machine']['DM'],
    #     "DMGated": "gated-ukdale-washing machine-DM_GATE2.csv"
    #     # "DMGated": UKDALE_DF_MAP['washing machine']['DMGated']
    # })
    #
    # best_window_wm = find_best_window(df_wash, "DMGated", 1200,
    #                                   utils.GENERAL_APP_META['washing machine'],
    #                                   step=200, sample_rate=6)
    #
    # print(df_wash.head())
    # plot_app("washing machine",
    #          df_wash,
    #          [
    #              "AttentionCNN",
    #              "SGN",
    #              "DM",
    #              "DMGated"],
    #          'ukdale', score=True,
    #          rng=best_window_wm)
    #
    # df_kettle = load_app("kettle", {
    #     "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
    #     "SGN": "ukdale-kettle-SGN.csv",
    #     "DM": "ukdale-kettle+fridge-DM_GATE2.csv",
    #     "DMGated": UKDALE_DF_MAP['dish washer']['DMGated'],
    # })
    #
    # best_window_kt = find_best_window(df_kettle, "DMGated", 500,
    #                                   utils.GENERAL_APP_META['kettle'],
    #                                   step=100, sample_rate=6, metric='mae')
    #
    # print("mw best window", best_window_kt)
    #
    # plot_app("kettle",
    #          df_kettle,
    #          [
    #              "AttentionCNN",
    #              "SGN",
    #              "DM",
    #              "DMGated"],
    #          'ukdale', score=True,
    #          rng=(best_window_kt[0] + 30, best_window_kt[1] + 30))

    # df_microwave = load_app("microwave", {
    #     "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
    #     "SGN": "ukdale-microwave-SGN.csv",
    #     # "DM": "ukdale-dish washer+washing machine+microwave-DM_GATE2.csv",
    #     "DMGated": UKDALE_DF_MAP['dish washer']['DMGated']
    # })
    # best_window_mw = find_best_window(df_microwave, "DMGated", 300,
    #                                   utils.GENERAL_APP_META['microwave'],
    #                                   step=100, sample_rate=6, metric='mae')
    #
    # print("mw best window", best_window_mw)
    #
    # plot_app("microwave",
    #          df_microwave,
    #          [
    #              "AttentionCNN",
    #              "SGN",
    #              # "DM",
    #              "DMGated"],
    #          'ukdale', score=True,
    #          rng=(best_window_mw[0] + 30, best_window_mw[1] + 30))


def draw_redd():
    # df_fridge = load_app("fridge", {
    #     "AttentionCNN": "redd-dish washer+washing machine+fridge-AttentionCNN.csv",
    #     "SGN": "redd-dish washer+washing machine+fridge-SGN.csv",
    #     "DM": "redd-fridge+washing machine+dish washer-DM_GATE2.csv",
    #     "DMGated": "gated-redd-fridge-DM_GATE2.csv"
    # })
    #
    # print(df_fridge.head())
    # plot_app("fridge", df_fridge,
    #          ["AttentionCNN", "SGN",
    #           "DM", "DMGated"
    #           ],
    #          'redd', score=True, rng=(0, 1000), filter_invalid=True)

    # df_dish = load_app("dish washer", {
    #     "AttentionCNN": "redd-dish washer+washing machine+fridge-AttentionCNN.csv",
    #     "SGN": "redd-dish washer+washing machine+fridge-SGN.csv",
    #     "DM": REDD_DF_MAP['washing machine']['DM'],
    #     "DMGated": REDD_DF_MAP['washing machine']['DMGated']
    # })
    #
    # print(df_dish.head())
    # plot_app("dish washer",
    #          df_dish,
    #          ["AttentionCNN", "SGN", "DM", "DMGated"],
    #          'redd', score=True, rng=(0, 1000))

    # df_wash = load_app("washing machine", {
    #     "AttentionCNN": "redd-dish washer+washing machine+fridge-AttentionCNN.csv",
    #     "SGN": "redd-dish washer+washing machine+fridge-SGN.csv",
    #     "DM": REDD_DF_MAP['washing machine']['DM'],
    #     "DMGated": REDD_DF_MAP['washing machine']['DMGated']
    # })
    #
    # print(df_wash.head())
    # plot_app("washing machine",
    #          df_wash,
    #          ["AttentionCNN", "SGN", "DM", "DMGated"],
    #          'redd', score=True, rng=(0, 1000))

    df_microwave = load_app("microwave", {
        "AttentionCNN": "redd-microwave-AttentionCNN.csv",
        "SGN": "redd-microwave-SGN.csv",
        "DM": "redd-microwave-DM_GATE2.csv",
        # "DM": REDD_DF_MAP['washing machine']['DM'],
        "DMGated": REDD_DF_MAP['washing machine']['DMGated']
    })

    print(df_microwave.head())
    plot_app("microwave",
             df_microwave,
             ["AttentionCNN", "SGN", "DM", "DMGated"],
             'redd',
             score=True, rng=(0, 1000))


def compute_score(app_name, clf_dict, filter_invalid=False):
    app = app_name
    clf_dfs = {}
    for k, v in clf_dict.items():
        if isinstance(v, str):
            df = load_app(app_name, {k: v})
            clf_dfs[k] = [df]
        elif isinstance(v, list) or isinstance(v, tuple):
            lst = []
            for vv in v:
                try:
                    df = load_app(app_name, {k: vv})
                    lst.append(df)
                except KeyError as e:
                    print("Cannot load from", k, vv)
                    raise e
            clf_dfs[k] = lst
    print(app_name)

    clf_scores = {}

    for clf, dfs in clf_dfs.items():
        scores = {'acc': [], 'F1': [], 'mae': [], 'sae': [], 'wssim': []}
        for df in dfs:
            truth = df.loc[:, 'truth'].to_numpy()
            pred = df.loc[:, clf].to_numpy()

            # valid_pred = filtering(pred, meta['on'], meta['max'])
            valid_truth = truth
            valid_pred = pred

            acc = losses.accuracy(app, truth, pred)
            f1 = losses.f1score(app, truth, pred)
            # pre = losses.precision(app, truth, pred)
            # recall = losses.recall(app, truth, pred)
            mae = losses.mae(app, truth, pred)
            sae = losses.sae(app, truth, pred, window_size=600)
            wssim = losses.wssim(app, valid_truth, valid_pred, window_size=11)

            scores['acc'].append(acc)
            scores['F1'].append(f1)
            scores['mae'].append(mae)
            scores['sae'].append(sae)
            scores['wssim'].append(wssim)

        sb = []
        for met, val_lst in scores.items():
            avg_val = sum(val_lst) / len(val_lst)
            sb.append(met)
            sb.append(f'{avg_val:.3f}' if avg_val > 1 else f'{avg_val:.4f}')
        print(clf, *sb)

        clf_scores[clf] = scores

    return clf_scores


def compute_avg_score(app_clf_score_list: list):
    clf_sum = {}
    for app_res in app_clf_score_list:
        if len(clf_sum) == 0:
            clf_sum.update({clf: {} for clf in app_res})
        for clf, scores in app_res.items():
            cs = clf_sum[clf]
            for metric, values in scores.items():
                avg_val = sum(values) / len(values)
                if metric in cs:
                    cs[metric] += avg_val
                else:
                    cs[metric] = avg_val

    n_apps = len(app_clf_score_list)
    for scores in clf_sum.values():
        for metric, value in scores.items():
            scores[metric] = value / n_apps
    return clf_sum


def compute_scores():
    wm = compute_score("washing machine",
                       {
                           "SGN": [
                               "../csv_ft/redd_seen_sgn_dish washer+fridge+washing machine_1_.csv",
                               "../csv_ft/redd_seen_sgn_dish washer+fridge+microwave+washing machine_3_.csv"],
                           "GT-NILM-GM": ["../csv_ft/washing machine_1_GATE=False_DM=False.csv",
                                          "../csv_ft/washing machine_3_GATE=False_DM=False.csv"],
                           "GT-NILM-AST": ["../csv_ft/washing machine_1_GATE=True_DM=False.csv",
                                           "../csv_ft/washing machine_3_GATE=True_DM=False.csv"],
                           "GH-NILM-1": ["../csv_ft/1days_washing machine_1_GATE=True_DM=True.csv",
                                         "../csv_ft/1days_washing machine_3_GATE=True_DM=True.csv"],
                           "GT-NILM-7": ["../csv_ft/washing machine_1_GATE=True_DM=True.csv",
                                         "../csv_ft/washing machine_3_GATE=True_DM=True.csv"]
                       })

    dw = compute_score("dish washer",
                       {
                           "SGN": [
                               "../csv_ft/redd_seen_sgn_dish washer+fridge+washing machine_1_.csv",
                               "../csv_ft/redd_seen_sgn_dish washer+fridge+microwave_2_.csv",
                               "../csv_ft/redd_seen_sgn_dish washer+fridge+microwave+washing machine_3_.csv"],
                           "GT-NILM-GM": ["../csv_ft/dish washer_1_GATE=False_DM=False.csv",
                                          "../csv_ft/dish washer_2_GATE=False_DM=False.csv",
                                          "../csv_ft/dish washer_3_GATE=False_DM=False.csv"],
                           "GT-NILM-AST": ["../csv_ft/dish washer_1_GATE=True_DM=False.csv",
                                           "../csv_ft/dish washer_2_GATE=True_DM=False.csv",
                                           "../csv_ft/dish washer_3_GATE=True_DM=False.csv"],
                           "GH-NILM-1": ["../csv_ft/1days_dish washer_1_GATE=True_DM=True.csv",
                                         "../csv_ft/1days_dish washer_2_GATE=True_DM=True.csv",
                                         "../csv_ft/1days_dish washer_3_GATE=True_DM=True.csv"],
                           "GT-NILM-7": ["../csv_ft/dish washer_1_GATE=True_DM=True.csv",
                                         "../csv_ft/dish washer_2_GATE=True_DM=True.csv",
                                         "../csv_ft/dish washer_3_GATE=True_DM=True.csv"]
                       })

    fr = compute_score("fridge",
                       {
                           "SGN": [
                               "../csv_ft/redd_seen_sgn_dish washer+fridge+washing machine_1_.csv",
                               "../csv_ft/redd_seen_sgn_dish washer+fridge+microwave_2_.csv",
                               "../csv_ft/redd_seen_sgn_dish washer+fridge+microwave+washing machine_3_.csv"],
                           "GT-NILM-GM": ["../csv_ft/fridge_1_GATE=False_DM=False.csv",
                                          "../csv_ft/fridge_2_GATE=False_DM=False.csv",
                                          "../csv_ft/fridge_3_GATE=False_DM=False.csv"],
                           "GT-NILM-AST": ["../csv_ft/fridge_1_GATE=True_DM=False.csv",
                                           "../csv_ft/fridge_2_GATE=True_DM=False.csv",
                                           "../csv_ft/fridge_3_GATE=True_DM=False.csv"],
                           "GH-NILM-1": ["../csv_ft/1days_fridge_1_GATE=True_DM=True.csv",
                                         "../csv_ft/1days_fridge_2_GATE=True_DM=True.csv",
                                         # "../csv_ft/1days_fridge_3_GATE=True_DM=True.csv"
                                         ],
                           "GT-NILM-7": ["../csv_ft/fridge_1_GATE=True_DM=True.csv",
                                         "../csv_ft/fridge_2_GATE=True_DM=True.csv",
                                         "../csv_ft/fridge_3_GATE=True_DM=True.csv"]
                       })

    mw = compute_score("microwave",
                       {
                           "SGN": ["../csv_ft/redd_seen_sgn_dish washer+fridge+microwave_2_.csv",
                                   "../csv_ft/redd_seen_sgn_dish washer+fridge+microwave+washing machine_3_.csv"],
                           "GT-NILM-GM": ["../csv_ft/microwave_2_GATE=False_DM=False.csv",
                                          "../csv_ft/microwave_3_GATE=False_DM=False.csv"],
                           "GT-NILM-AST": ["../csv_ft/microwave_2_GATE=True_DM=False.csv",
                                           "../csv_ft/microwave_3_GATE=True_DM=False.csv"],
                           "GH-NILM-1": ["../csv_ft/1days_microwave_2_GATE=True_DM=True.csv",
                                         "../csv_ft/1days_microwave_3_GATE=True_DM=True.csv"
                                         ],
                           "GT-NILM-7": ["../csv_ft/microwave_2_GATE=True_DM=True.csv",
                                         "../csv_ft/microwave_3_GATE=True_DM=True.csv"]
                       })

    clf_avg_scores = compute_avg_score([dw, fr, mw, wm])
    # print(clf_avg_scores)

    print("===== average =====")
    for clf, met_vals in clf_avg_scores.items():
        sb = []
        for met, val in met_vals.items():
            sb.append(met)
            sb.append(f'{val:.3f}' if val > 1 else f'{val:.4f}')
        print(clf, *sb)


def compute_method_averages(csv_file):
    """
    Reads a CSV file containing NILM performance metrics and computes
    the average of each method for each metric.

    Parameters:
    - csv_file (str): Path to the CSV file.

    Returns:
    - A DataFrame containing the average values for each method.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Remove rows where 'Method' is NaN (likely empty rows)
    df = df.dropna(subset=['Method'])

    # Identify numeric columns (metrics)
    metric_columns = ['Acc. ↑', 'F1-score ↑', 'MAE ↓', 'SAE ↓', 'WSSIM ↑']

    # Convert all numeric columns to float, handling missing values ('----')
    for col in metric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute the average for each method
    method_averages = df.groupby('Method')[metric_columns].mean()

    return method_averages


def compute_csv_average():
    # Example usage
    csv_path = "metrics/scores.csv"  # Replace with the actual path to your CSV file
    method_averages = compute_method_averages(csv_path)

    print(method_averages)

    # # Display results
    # import ace_tools as tools
    # tools.display_dataframe_to_user(name="Method Averages", dataframe=method_averages)


def draw_transfer():
    # plot_app_gate("microwave",
    #               "./csv_ft/microwave_3_GATE=True_DM=True.csv",
    #               (27500, 28500))
    plot_app_transfer_multi("washing machine",
                            {
                                "SGN": "../csv_ft/redd_seen_sgn_dish washer+fridge+washing machine_1_.csv",
                                "DMGated": "../csv_ft/washing machine_1_GATE=True_DM=True.csv"
                            },
                            (69150, 70800))
    plot_app_transfer_multi("dish washer",
                            {
                                "SGN": "../csv_ft/redd_seen_sgn_dish washer+fridge+washing machine_1_.csv",
                                "DMGated": "../csv_ft/dish washer_1_GATE=True_DM=True.csv"
                            },
                            (9500, 11000)
                            # (34500, 36000)
                            )


def draw_bar():
    df = pd.read_csv("./metrics/time.csv")
    print(df.head())
    res = {}

    for app in APP_NAMES:
        model = df[df['app'] == app]
        times = list(model['total_time'].values)
        res[app] = times

    # for model in mapp:
    #     apps = df[df['model'] == model]
    #     name = model
    #     times = list(apps['total_time'].values)
    #     res[mapp[name]] = times

    print(res)

    order = ['Refrigerator', 'Dish washer', 'Washing machine', 'Microwave oven', 'Kettle']

    plt.figure(figsize=(12, 6))

    # colors = ['#e9824d', '#d6d717', '#92cad1', '#868686', '#79ccb3']
    # colors = ['#4e669e', '#699b87', '#86ac51', '#fcc602', '#f0df72']
    colors = ['#92C9C0', '#F8F4BD', '#C1BED5', '#EF9388', '#868686']
    hatches = ['/', '//', '--', 'x', '+']

    x = np.arange(4)
    pos = -1 / 3
    i = 0
    for name, values in res.items():
        xs = x + pos
        plt.bar(xs, values, 1 / 6, align='center', color=colors[i], hatch=hatches[i])
        pos += 1 / 6
        i += 1

    plt.xticks(x, ['Attention CNN', 'SGN', 'U-Net-DM', 'GT-NILM$^\mathrm{GM}$'])
    plt.xlabel("Method")
    plt.ylabel("Training time (seconds)")
    plt.legend([APP_NAMES[app] for app in APP_NAMES])
    plt.tight_layout()
    plt.savefig(f"./figures/time_cmp.svg", bbox_inches='tight')
    plt.show()


def draw_time_cmp():
    df = pd.read_csv("./metrics/time.csv")
    print(df.head())

    # Mapping of model names from CSV to display names
    model_map = {
        'sgn': 'SGN',
        'attn': 'Attention CNN',
        'dm': 'U-Net-DM',
        'dm-gate': 'GT-NILM$^\mathrm{GM}$'
    }

    order = ['fridge', 'dish washer', 'washing machine', 'microwave', 'kettle']

    res = {}
    inf_time = {}

    # Extract training times
    for app in order:
        model = df[df['app'] == app]
        times = list(model['total_time'].values)
        res[app] = times

    # Extract inference time (from washing machine * 4)
    for model_key in model_map:
        model_display = model_map[model_key]
        inf_time[model_display] = \
            df[(df['app'] == 'washing machine') & (df['model'] == model_key)]['inf_time'].values[
                0] * 5 / 5

    print("Training Time:", res)
    print("Total Inference Time:", inf_time)

    plt.figure(figsize=(12, 6))

    colors = ['#92C9C0', '#F8F4BD', '#C1BED5', '#EF9388', '#868686']
    hatches = ['/', '//', '--', 'x', '+']
    inf_color = '#333333'  # Dark gray/black for inference bars

    x = np.arange(len(model_map)) * 1.15  # Increased spacing between method groups
    bar_width = 1 / (len(order) + 2)  # Adjusted for proper spacing

    pos = -len(order) / 2 * bar_width  # Center-align training bars
    i = 0

    # Create Primary Y-axis for Training Time
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot training time bars
    bars = []
    for name, values in res.items():
        xs = x + pos
        bars.append(
            ax1.bar(xs, values, bar_width, align='center', color=colors[i], hatch=hatches[i],
                    label=name))
        pos += bar_width
        i += 1

    # Set Primary Y-axis Label (Training Time)
    ax1.set_ylabel("Training time (seconds)")

    # Create Secondary Y-axis for Inference Time
    ax2 = ax1.twinx()
    ax2.set_ylabel("Total inference time (seconds)")

    # Correct Inference Bar Placement (aligned with training bars)
    inf_xs = x + pos + (bar_width / 2)  # Small offset for better alignment
    inf_ys = [inf_time[model_map[m]] for m in model_map]
    inf_bar = ax2.bar(inf_xs, inf_ys, bar_width, align='center', color=inf_color,
                      label="Total inference time")

    # Set X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels([model_map[m] for m in model_map])
    ax1.set_xlabel("Method")

    # Fixing the Legend (Ensure it contains appliances + inference time)
    legend_entries = [bar[0] for bar in bars]  # Get first element from each appliance bar group
    legend_labels = [APP_NAMES[ap] for ap in order]  # Appliance names
    legend_entries.append(inf_bar[0])  # Add inference bar
    legend_labels.append("Total inference time")  # Add inference label

    ax1.legend(legend_entries, legend_labels, loc='upper left')

    plt.tight_layout()
    plt.savefig("./figures/time_cmp_fixed.svg", bbox_inches='tight')
    plt.show()


def plot_loss(path, plt_index, title=None):
    df = pd.read_csv(path)
    df = df.dropna()

    # plt.subplot(1, 2, plt_index)

    x = df['epoch'].values
    train_loss = df['train_loss'].values
    val_loss = df['val_loss'].values

    # plt.plot(x, train_loss, label='Train Loss')
    plt.plot(x, val_loss, label='Validation Loss')
    min_index = np.argmin(val_loss)
    print(val_loss, min_index)
    plt.axvline(x=min_index, color='orange')

    # if plt_index == 1:
    plt.ylabel('Loss')
    plt.xlabel('Epoch index')

    if title is not None:
        plt.title(title)

    # plt.legend()


def plot_losses():
    plt.figure(figsize=(4, 3))
    # fig.tight_layout(pad=5.0)
    # plot_loss('./training_logs/dish_ukdale_dm/loss.csv', 1, 'U-Net-DM')
    plot_loss('./training_logs/dish_ukdale_dm_g2/loss.csv', 1)
    plt.tight_layout()

    # plt.subplots_adjust(wspace=0.25, bottom=0.15)

    plt.savefig(f"./figures/loss.svg", bbox_inches='tight')
    plt.show()


def filter_invalid_data(data, on, max_value):
    """
    Filters out invalid data by:
    - Setting values lower than `on` to 0.
    - Setting values higher than `max_value` to `max_value`.

    Parameters:
    - data (numpy array, pandas DataFrame, or pandas Series): Input data.
    - on (float): Lower threshold; values below this will be set to 0.
    - max_value (float): Upper threshold; values above this will be set to `max_value`.

    Returns:
    - Filtered data (same type as input).
    """
    data = np.where(data < on, 0, data)  # Set values below "on" to 0
    data = np.where(data > max_value, max_value, data)  # Cap values at "max_value"
    return data


def compute_similarities(app_name, df, methods):
    truth = df['truth'].values

    app_meta = utils.GENERAL_APP_META[app_name]
    truth = filter_invalid_data(truth, app_meta["on"], app_meta["max"])
    print("Truth min max", min(truth), max(truth))

    method_scores = {method: {} for method in methods}

    for method in methods:
        data = df[method].values
        data = filter_invalid_data(data, app_meta["on"], app_meta["max"])
        # print(method, "min max", min(data), max(data))

        # ssim = compute_ssim_1d(data, truth)
        # ssim = structural_similarity(data, truth, data_range=max(truth))
        sae = losses.sae("", truth, data)
        ssim = structural_similarity(truth, data, win_size=11, data_range=max(truth))
        ms_ssim_ = ms_ssim_1d(truth, data, 4, win_size=11)
        hfe = compute_hfe(truth, data)
        wd = wasserstein_distance(truth, data)
        psnr = compute_psnr(truth, data)
        snr = compute_snr(truth, data)
        gds = compute_gradient_difference_similarity(truth, data)
        scores = {"SAE": sae, "SSIM": ssim, "MS-SSIM": ms_ssim_, "HFE": hfe, "WD": wd,
                  "SNR": snr, "PSNR": psnr,
                  "GDS": gds}
        method_scores[method] = scores
        print(method, scores)

    return method_scores


def compute_ssims(ds, apps, methods):
    sims = {}
    for app in apps:
        df = load_app_ds(app, ds)
        print("=====", app, "=====")
        sim = compute_similarities(app, df, methods)
        sims[app] = sim

    print("===== Average =====")
    metrics = set()
    for v1 in sims.values():
        for v2 in v1.values():
            metrics = v2.keys()

    method_metric_scores = {}
    for method in methods:
        method_metric_scores[method] = {}
        for metric in metrics:
            metric_sum = 0
            for app in apps:
                method_metric = sims[app][method][metric]
                metric_sum += method_metric
            method_metric_scores[method][metric] = metric_sum / len(apps)

    for method, metric in method_metric_scores.items():
        print(method, metric)


def compute_hfe(real_signal, generated_signal):
    """
    Compute High-Frequency Emphasis (HFE) based on Fourier Transform.
    """
    real_fft = np.fft.fft(real_signal)
    gen_fft = np.fft.fft(generated_signal)

    high_freq_real = np.abs(real_fft[len(real_fft) // 4:])  # Focus on high frequencies
    high_freq_gen = np.abs(gen_fft[len(gen_fft) // 4:])

    return np.linalg.norm(high_freq_real - high_freq_gen) / np.linalg.norm(high_freq_real)


def downsample(signal):
    """
    Apply a Gaussian filter for anti-aliasing and downsample by a factor of 2.
    """
    # Use Gaussian smoothing to reduce aliasing effects.
    smoothed = gaussian_filter1d(signal, sigma=1)
    return smoothed[::2]


def ms_ssim_1d(truth, pred, levels=4, weights=None, win_size=11):
    """
    Compute the MS-SSIM for 1D signals.

    Parameters:
      signal1, signal2 : 1D numpy arrays of equal length.
      levels         : Number of scales. For a 600-sample signal, 4 levels is often reasonable.
      weights        : List or array of weights (length equal to `levels`).
                       If None, a default weight vector is used.
      win_size       : Window size used for the SSIM computation at the original scale.

    Returns:
      ms_ssim_score : The final MS-SSIM score computed as a weighted geometric mean.
    """
    rng = max(truth)
    # Define default weights if not provided.
    if weights is None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        weights = weights / weights.sum()  # Ensure weights sum to 1.
    else:
        weights = np.array(weights)
        if len(weights) != levels:
            raise ValueError("Length of weights must equal the number of levels.")

    mssim_values = []

    # Copy the signals to avoid modifying the originals.
    s1 = truth.copy()
    s2 = pred.copy()

    for level in range(levels):
        # Compute SSIM directly on the 1D signals.
        ssim_val = structural_similarity(s1, s2, win_size=win_size, data_range=rng,
                                         gaussian_weights=True, use_sample_covariance=False)
        mssim_values.append(ssim_val)

        # Downsample for the next scale except for the last level.
        if level < levels - 1:
            s1 = downsample(s1)
            s2 = downsample(s2)

    mssim_values = np.array(mssim_values)
    # Combine the SSIM values using a weighted geometric mean.
    ms_ssim_score = np.prod(mssim_values ** weights)
    return ms_ssim_score


def compute_signal_psnr(truth, pred, window):
    psnr_sum = 0

    n_seg = len(truth) // window
    active = 0
    for i in range(n_seg):
        idx = i * window
        gt = truth[idx:idx + window]
        pr = pred[idx:idx + window]
        if np.any(gt):
            active += 1
            psnr = compute_psnr(gt, pr)
            # if np.isnan(psnr) or np.isinf(psnr):
            #     psnr = 0
            psnr_sum += psnr

    return psnr_sum / active


def compute_psnr(real_signal, generated_signal):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    """
    # real_signal, generated_signal = min_max_normalize(real_signal), min_max_normalize(
    #     generated_signal)
    mse = np.mean((real_signal - generated_signal) ** 2)
    max_signal = np.max(real_signal)
    return 20 * np.log10(max_signal / np.sqrt(mse))


def compute_snr(real_signal, generated_signal):
    """
    Compute Signal-to-Noise Ratio (SNR) in decibels (dB).
    Higher SNR means better signal quality.
    """
    # Compute signal power (sum of squared real signal)
    signal_power = np.sum(real_signal ** 2)

    # Compute noise power (sum of squared differences)
    noise_power = np.sum((real_signal - generated_signal) ** 2)

    if noise_power == 0:
        return float('inf')  # Perfect match, SNR is infinite

    # Compute SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)

    return snr


def compute_gradient_difference_similarity(real_signal, generated_signal, epsilon=1e-8):
    """
    Compute Gradient Difference Similarity (GDS) to measure shape and edge preservation.
    Higher means better similarity.
    """
    real_signal, generated_signal = min_max_normalize(real_signal), min_max_normalize(
        generated_signal)
    grad_real = np.diff(real_signal)
    grad_gen = np.diff(generated_signal)

    # Compute gradient difference
    grad_diff = np.abs(grad_real - grad_gen)

    # Compute similarity
    return 1 - (np.linalg.norm(grad_diff) / (np.linalg.norm(grad_real) + epsilon))


def min_max_normalize(signal):
    """
    Normalize a 1D signal using Min-Max Scaling to range [0,1].
    """
    return (signal - np.min(signal)) / (
            np.max(signal) - np.min(signal) + 1e-8)  # Avoid division by zero


if __name__ == '__main__':
    font = {'family': 'Times New Roman',
            'weight': 'normal'}
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 22}
    matplotlib.rc('font', **font)
    # matplotlib.rcParams['hatch.linewidth'] = 1.0
    # draw_direct()
    draw_ukdale_real_time()
    # draw_redd()
    # draw_transfer()

    # compute_scores()
    # compute_csv_average()

    # draw_bar()
    # draw_time_cmp()
    # plot_losses()

    # compute_ssims("ukdale",
    #               ["fridge", "dish washer",
    #                "washing machine", "kettle", "microwave"
    #                ],
    #               ["AttentionCNN", "SGN", "DM", "DMGated"])
