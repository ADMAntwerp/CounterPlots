import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


def make_greedy_plot(factual_score, features_data, class_names, save_path):
    fig = plt.figure(figsize=(7, 1 * len(features_data)))

    scatter_points = [f['score'] for f in features_data]
    scatter_points.insert(0, factual_score)

    cmap = ["#8f8f8f", "#d12771", "#4589ff", "#007d79", "#8a3ffc", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5",
            "#8bd3c7"]
    markers = ['+', 'o', '^', 's', 'p', 'P', '*', 'X', 'D']
    cmark = ['●', '▲', '■', '⬟', '✚', '✖', '★', '◆']

    plt.plot(scatter_points, range(len(scatter_points)),
             color='#c4c4c4', linestyle='dashed', zorder=0)
    for c_idx, point in enumerate(scatter_points):
        plt.scatter(
            [point],
            [c_idx],
            marker=markers[c_idx],
            color=cmap[c_idx],
            s=100,
            edgecolors='#545454' if c_idx != 0 else None,)

    features_names = ['Factual']
    for feat_idx, f in enumerate(features_data):
        features_names.append(
            f'{cmark[feat_idx]} - {f["name"]} ({f["factual"]}➜{f["counterfactual"]})')
    max_feat_names_length = max([len(feat_name)
                                for feat_name in features_names])

    # Plot a vertical line at the point of the highest score
    plt.axvline(x=0.5, color='#c20000', linestyle='dashed', zorder=0)

    for feat_idx, feat_name in enumerate(features_names):
        plt.text(-0.02 * max_feat_names_length, feat_idx,
                 feat_name, color=cmap[feat_idx], fontsize=12)
        if feat_idx > 1:
            plt.text(-0.02 * max_feat_names_length, feat_idx - 0.35, f'+{",".join(cmark[:feat_idx - 1])}',
                     color='#8f8f8f', fontsize=12, bbox=dict(facecolor='none', edgecolor='#8f8f8f'))

    # Print binary class
    plt.text(0.5 - 0.14 * len(list(class_names.values())[0]) / 8, len(features_names)*1.025 - 0.9,
             list(class_names.values())[0], color='#ff0055D2', fontweight='bold')
    plt.text(0.49, len(features_names)*1.025 - 0.9, f'➜',
             color='#c20000', fontweight='bold')
    plt.text(0.52, len(features_names)*1.025 - 0.9, list(class_names.values())
             [1], color='#008ae7D2', fontweight='bold')

    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Set limit of x axis from 0 to 1
    plt.xlim(-0.05, 1)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


def make_countershapley_plot(factual_score, features_data, classes, save_path):
    fig = plt.figure(figsize=(10, 1.5))
    ax = fig.add_subplot(111)
    scale_x = 100
    scale_y = 100
    bar_y_height = scale_y / 2
    fontsize = 10
    color_factual = '#ff0055D2'
    color_counterfactual = '#008ae7D2'

    max_score = features_data[-1]['score']

    x_threshold = (0.5 - factual_score) / \
        (features_data[-1]['score'] - factual_score)*100

    # Draw bar for the factual score
    plt.bar(0, scale_y - 10, width=0.5, color='#ff0055', linewidth=1)
    plt.text(-5, scale_y + fontsize * 4, 'Factual Score',
             color='#ff0055', fontsize=fontsize)
    plt.text(0, scale_y, factual_score, color='#ff0055',
             fontsize=fontsize, fontweight='bold')

    # Print class names
    size_factual_class = mpl.textpath.TextPath(
        (0, 0), classes[0], size=fontsize).get_extents().width * 0.2037 + 1.090
    plt.text(x_threshold - size_factual_class, scale_y + fontsize * 7, f'{classes[0]}', color='#ff0055D2',
             fontsize=fontsize, fontweight='bold')
    plt.text(x_threshold - 1, scale_y + fontsize * 7, f'➜',
             color='#c20000', fontsize=fontsize, fontweight='bold')
    plt.text(x_threshold + 1, scale_y + fontsize * 7, f'{classes[1]}', color='#008ae7D2', fontsize=fontsize,
             fontweight='bold')

    # Beak must be always the 4%
    def create_bar(start_x, end_x, no_beak=False):

        verts = [
            (start_x, 0.),  # left, bottom
            (start_x, bar_y_height),  # left, top
            (end_x - 4, bar_y_height),  # right, top
            (end_x, bar_y_height),  # right, top
            (end_x, 0.),  # right, bottom
            (0., 0.),  # ignored
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO if no_beak else Path.CURVE3,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        return Path(verts, codes)

    # Bar for feature names and feature changes
    plt.bar(0, -scale_y + 10, width=200, color='#ebebeb', linewidth=0)

    current_x = 0
    x_left_pos = []
    for f_idx, feat_data in enumerate(features_data):
        x_left_pos.append(current_x)
        x_size = feat_data['x'] * 100 / scale_x + current_x

        # Plot text for feature name
        plt.text(
            current_x + (x_size - current_x) / 2 -
            len(feat_data['name']) * fontsize / 10 / 2,
            -30,
            feat_data['name'],
            color='#545454',
            fontsize=fontsize)

        feature_change_text = f"{feat_data['factual']}➜{feat_data['counterfactual']}"

        # Plot bar up to 50% of the plot with the factual color
        if x_size < x_threshold:
            ax.add_patch(patches.PathPatch(create_bar(
                current_x, x_size), facecolor=color_factual, lw=0))
        else:
            if current_x < x_threshold:
                ax.add_patch(
                    patches.PathPatch(create_bar(current_x, x_threshold, no_beak=True), facecolor=color_factual, lw=0))
                ax.add_patch(patches.PathPatch(create_bar(
                    x_threshold, x_size), facecolor=color_counterfactual, lw=0))
            else:
                ax.add_patch(patches.PathPatch(create_bar(
                    current_x, x_size), facecolor=color_counterfactual, lw=0))

        # Plot text for feature changes
        feat_change_text = mpl.textpath.TextPath(
            (0, 0), feature_change_text, size=fontsize)
        plt.text(
            current_x + (x_size - current_x)/2 -
            feat_change_text.get_extents().width * 0.1,
            -70,
            feature_change_text,
            color='#545454',
            fontsize=fontsize)

        current_x = x_size

        # Print score bold
        plt.text(
            current_x,
            scale_y,
            feat_data['score'],
            color='#008ae7' if f_idx == len(features_data) - 1 else '#999999',
            fontsize=fontsize,
            fontweight='bold')

    # Plot bar for the counterfactual score
    plt.bar(current_x, scale_y - 10, width=0.5, color='#008ae7', linewidth=1)
    plt.text(current_x - 10, scale_y + fontsize * 4,
             'Counterfactual Score', color='#008ae7', fontsize=fontsize)

    # Draw bar for threshold
    plt.bar(x_threshold, scale_y - 10, width=0.25,
            color='#c20000', linewidth=1)

    # Remove y axis
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    # Remove square around the plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    for l_pos in x_left_pos:
        plt.bar(l_pos, -scale_y + 10, width=0.2, color='#cccccc', linewidth=0)

    ax.set_xlim(0, scale_x)
    ax.set_ylim(-scale_y, scale_y + 70)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def make_constellation_plot(factual_score, single_points_chart, text_features, mulitple_points_chart,
                            mulitple_points_chart_y, single_points, class_names, cf_score, point_to_pred, save_path):
    x_dim = 10
    y_dim = 4
    y_lim_low = -0.1
    y_lim_high = len(single_points_chart) - 0.9
    x_lim_low = 0
    x_lim_high = 1
    cmap = ["#d12771", "#4589ff", "#007d79", "#8a3ffc",
            "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]

    fig = plt.figure(figsize=(x_dim, y_dim))
    ax = fig.add_subplot(111)

    corr_x_dim = (x_lim_high - y_lim_low) / x_dim
    corr_y_dim = (y_lim_high - y_lim_low) / y_dim

    ax.set_xlim(x_lim_low, x_lim_high)
    ax.set_ylim(y_lim_low, y_lim_high)

    # Plot single change points
    for idx_p, p in enumerate(single_points_chart):
        plt.scatter([p[1]], [p[0]], color=cmap[idx_p], s=100)

    # Plot feature names and value changes
    max_text_features = max([len(f) for f in text_features])
    for i in range(len(text_features)):
        plt.text(-0.012*max_text_features, i,
                 text_features[i], color=cmap[i], fontsize=12)

    # Verify if there are multiple change points
    if len(mulitple_points_chart) > 0:
        # Plot multiple change points
        plt.scatter(mulitple_points_chart[:, 1],
                    mulitple_points_chart_y, color='blue', s=10)

    # Plot counterfactual point
    cf_pred_x_1 = np.mean([*range(len(single_points))])
    plt.scatter([cf_score], [cf_pred_x_1], color='#FFB449', s=100)

    # Plot a vertical line at the factual score
    plt.axvline(x=factual_score, color='#8f8f8f', linestyle='dashed', zorder=0)
    plt.text(factual_score, len(text_features) - 1.2, 'Factual',
             rotation='vertical', color='#8f8f8f', fontsize=10)

    # Plot a vertical line at the threshold
    plt.axvline(x=0.5, color='#c20000', linestyle='dashed', zorder=0)

    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Plot classes names
    plt.text(0.5 - 0.10 * len(class_names[0]) / 8,
             len(text_features)*1.01 - 0.9, class_names[0], color='#ff0055D2', fontweight='bold')
    plt.text(0.49, len(text_features)*1.01 - 0.9, f'➜',
             color='#c20000', fontweight='bold')
    plt.text(0.51, len(text_features)*1.01 - 0.9,
             class_names[1], color='#008ae7D2', fontweight='bold')

    # Plot Counterfactual lines
    for i in range(len(single_points)):
        x_0 = point_to_pred[i]
        x_1 = cf_score
        y_0 = i
        y_1 = cf_pred_x_1
        plt.plot([x_0, x_1], [y_0, y_1], color='k', zorder=0,
                 linewidth=1, alpha=0.15, linestyle='dotted')

    for points, x_value in mulitple_points_chart:
        # plt.plot([points[0], point_to_pred[points[0]]], [1, 1], color='blue')
        # [x1, x2], [y1, y2]
        for origin_point in points:
            x_0 = point_to_pred[origin_point]
            x_1 = x_value
            y_0 = origin_point
            y_1 = np.mean(points)
            # Slope-Intercept formula
            # f = lambda x: (y_1 - y_0) / (x_1 - x_0) * x + y_0 - (x_0 * (y_1 - y_0) / (x_1 - x_0))

            # angle_arrow = np.arctan((y_1-y_0)/((x_1-x_0)))
            # cos_arrow = np.cos(angle_arrow)
            # sin_arrow = np.sin(angle_arrow)

            # # Euclidean distance between x and y
            # dist_x_y = np.sqrt((x_1-x_0)**2 + (y_1-y_0)**2)

            # print(40*cos_arrow/dist_x_y)
            # ax.quiver((x_0+x_1)/2, (y_0+y_1)/2, cos_arrow*corr_y_dim, sin_arrow*corr_x_dim, width=0.005, color='#e0e0e0', zorder=0, pivot='tip')
            plt.plot([x_0, x_1], [y_0, y_1], color='#e0e0e0',
                     zorder=0, linewidth=1, alpha=0.5)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
