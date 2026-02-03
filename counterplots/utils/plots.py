import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

PINK1 = "#ff0055"
PINK1_TR = "#ff0055D2"
RED1 = "#c20000"
BLUE1 = "#008ae7"
BLUE_TR = "#008ae7D2"
DARK_GREY = "#545454" 
LIGHT_GREY = "#e0e0e0" 


def _get_text_width_in_data_co(text, fig, ax):
    renderer = fig.canvas.get_renderer()
    fig.canvas.draw()

    bbox = text.get_extents()
    x0, _ = ax.transData.inverted().transform((bbox.x0, bbox.y0))
    x1, _ = ax.transData.inverted().transform((bbox.x1, bbox.y0))

    width = x1 - x0
    return width


def _infer_text_orientation(text, fig, ax, w_limit):
    if _get_text_width_in_data_co(text, fig, ax) > w_limit: return "vertical" 
    return "horizontal"

def make_greedy_plot(factual_score, features_data, class_names, save_path):
    fig = plt.figure(figsize=(7, 1 * len(features_data))) # noqa

    scatter_points = [f['score'] for f in features_data]
    scatter_points.insert(0, factual_score)

    plt.plot(
        scatter_points, 
        range(len(scatter_points)),
        color='#c4c4c4', 
        linestyle='dashed', 
        zorder=0
        )
    for c_idx, point in enumerate(scatter_points):
        plt.scatter(
            [point],
            [c_idx],
            color=PINK1_TR if point < 0.5 else BLUE_TR,
            s=100)

    features_names = ['Factual']
    for feat_idx, f in enumerate(features_data):
        features_names.append(
            f'{f["name"]} ({f["factual"]}➜{f["counterfactual"]})')
    max_feat_names_length = max([len(feat_name) for feat_name in features_names])

    # Plot a vertical line at the threshold score
    plt.axvline(x=0.5, color=RED1, zorder=0, linewidth=0.5)

    # Plot a vertical line at the factual score
    plt.axvline(x=factual_score, color=PINK1_TR,
                linestyle='dashed', zorder=0)
    plt.text(factual_score - 0.088, len(features_names)*1.04 -
             0.9, 'Factual Score', color=PINK1_TR)

    # Plot a vertical line at the counterfactual score
    plt.axvline(x=scatter_points[-1],
                color=BLUE_TR, linestyle='dashed', zorder=0)
    plt.text(scatter_points[-1] - 0.15, len(features_names)*1.04 -
             0.9, 'Counterfactual Score', color=BLUE_TR)

    for feat_idx, feat_name in enumerate(features_names):
        plt.text(-0.02 * max_feat_names_length, feat_idx,
                 feat_name, color=DARK_GREY, fontsize=12)

    # Print binary class
    class_0_name = list(class_names.values())[0]
    class_1_name = list(class_names.values())[1]
    size_factual_class = mpl.textpath.TextPath(
        (0, 0), class_0_name).get_extents().width * 0.003
    plt.text(0.47 - size_factual_class - max([len(f) for f in features_names])*0.0001, len(features_names)
             - 0.75 + (len(features_names) + 1)*0.04,
             class_0_name, color=PINK1_TR, fontweight='bold')
    plt.text(0.49, len(features_names) - 0.75 + (len(features_names) + 1)*0.04, '➜',
             color=RED1, fontweight='bold')
    plt.text(0.52, len(features_names) - 0.75 + (len(features_names) + 1)
             * 0.04, class_1_name, color=BLUE_TR, fontweight='bold')

    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Set limit of x axis from 0 to 1
    plt.xlim(-0.05, 1)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


def make_countershapley_plot(factual_score, features_data, classes, save_path):
    """

    Args:
        factual_score (float): The model score for the factual instance
        features_data (List[Dict]): List of dictionaries, each dictionary containing info on a CounterShapley value.
            Each dictionary must contain the keys:
            
            - `x`: the CounterShapley value itself
            - 'score': the (non-greedy) model prediction score associated with this countershapley value
            - `name`: the name of the feature
            - `factual`: the factual value of the feature
            - `counterfactual`: the counterfactual value of the feature

        classes (unused)
        save_path (str): A path to save the plot to
    """
    fig = plt.figure(figsize=(10, 1.5))
    ax = fig.add_subplot(111)
    scale_x = 100
    scale_y = 100
    bar_y_height = scale_y / 2
    bar_y_height_sub = scale_y / 8.0
    fontsize = 10
    color_bar = '#48C975D2'
    cf_score = features_data[-1]['score']
    sum_countershapley = cf_score - factual_score

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

    prev_score = factual_score
    current_x = 0
    x_left_pos = []
    for _, feat_data in enumerate(features_data):
        x_left_pos.append(current_x)
        x_size = feat_data['x'] * 100 / scale_x + current_x

        # Plot text for feature name
        feature_change_text = f"{feat_data['name']}\n{feat_data['factual']}➜{feat_data['counterfactual']}"
        feat_change_textpath = mpl.textpath.TextPath((0, 0), feature_change_text, size=fontsize)
        feat_cs_score_percentage = round((feat_data['score'] - prev_score)/sum_countershapley*100, 1)
        feat_cs_score_percentage_text = mpl.textpath.TextPath((0, 0), f'{feat_cs_score_percentage}%', size=fontsize)

        rotation = _infer_text_orientation(
            text=feat_change_textpath, 
            fig=fig, 
            ax=fig.gca(), 
            w_limit=(x_end-current_x)/scale_x)

        txt_h = -40 if yscale > 0 else -40 - bar_y_height
        if rotation == "vertical":
            txt_h += 40  # offset "anchor" alignment
            ha="right"
        else: 
            ha="center"

        plt.text(
            current_x + (x_end - current_x)/2,
            txt_h,
            feature_change_text,
            color=DARK_GREY,
            fontsize=fontsize,
            ha=ha,
            va="center",
            rotation=rotation,
            rotation_mode="anchor"  # first align, then rotate
        )
        text_y_height = bar_y_height_sub if yscale > 0 else - bar_y_height + bar_y_height_sub
        plt.text(
            current_x + (x_end - current_x) / 2 -
            feat_cs_score_percentage_text.get_extents().width * 0.085,
            text_y_height,
            f'{feat_cs_score_percentage}%',
            color='white',
            fontsize=fontsize,
            weight="bold")

        current_x = x_size
        prev_score = feat_data['score']

    # Print CF score
    plt.text(
        current_x,
        scale_y,
        round(features_data[-1]['score'], 2),
        color=BLUE1,
        fontsize=fontsize)

    # Draw bar for the factual score
    plt.bar(0, scale_y - 10, width=0.5, color=PINK1, linewidth=1)
    plt.text(-5, scale_y + fontsize * 4, 'Factual Score',
             color=PINK1, fontsize=fontsize)
    plt.text(0, scale_y, factual_score, color=PINK1,
             fontsize=fontsize)

    # Plot bar for the counterfactual score
    plt.bar(current_x, scale_y - 10, width=0.5, color=BLUE1, linewidth=1)
    plt.text(current_x - 10, scale_y + fontsize * 4,
             'Counterfactual Score', color=BLUE1, fontsize=fontsize)

    # Remove y axis
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    # Remove square around the plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    for l_pos in x_left_pos:  # little dividers between x labels
        plt.bar(l_pos, -scale_y + 10, width=0.2, color='#cccccc', linewidth=0)
    plt.bar(scale_x, -scale_y + 10, width=0.2,
            color='#cccccc', linewidth=0)  # divider at the end

    ax.set_xlim(0, scale_x)
    ax.set_ylim(-scale_y, scale_y + 70)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def make_constellation_plot(
    factual_score, 
    single_points_chart, 
    text_features, 
    mulitple_points_chart,
    mulitple_points_chart_y, 
    single_points, 
    class_names, 
    cf_score, 
    point_to_pred, 
    save_path,
    highlight_feature=None
    ):
    if highlight_feature is not None: assert isinstance(highlight_feature, int), "Please pass a feature index (int) for the keyword `highlight`"
    x_dim = 10
    y_dim = 4
    y_lim_low = -0.1
    y_lim_high = len(single_points_chart) - 0.9
    x_lim_low = 0
    x_lim_high = 1

    fig = plt.figure(figsize=(x_dim, y_dim))
    ax = fig.add_subplot(111)

    ax.set_xlim(x_lim_low, x_lim_high)
    ax.set_ylim(y_lim_low, y_lim_high)

    # Plot single change points
    for idx_p, p in enumerate(single_points_chart):
        adapted_feat_ind, score = p
        color = BLUE_TR if score >= 0.5 else PINK1_TR
        if highlight_feature is not None and highlight_feature != adapted_feat_ind: color=LIGHT_GREY
        plt.scatter(
            [score], 
            [adapted_feat_ind], 
            color=color, 
            s=100
            )

    # Plot feature names and value changes
    max_text_features = max([len(f) for f in text_features])
    for i in range(len(text_features)):
        plt.text(
            -0.012*max_text_features, 
            i,
            text_features[i], 
            color=DARK_GREY, 
            fontsize=12
            )

    # Verify if there are multiple change points
    if len(mulitple_points_chart) > 0:
        adapted_feat_inds_per_point = mulitple_points_chart[:, 0]  # average height ~ average adapted feat inds.
        scores = mulitple_points_chart[:, 1]
        colors = [
            '#E0423A' if s < 0.5 and (highlight_feature is None or highlight_feature in f_ind)
            else LIGHT_GREY if s < 0.5 and highlight_feature not in f_ind
            else 'blue' 
            for s, f_ind in zip(scores, adapted_feat_inds_per_point)
        ]
        plt.scatter(
            scores,
            mulitple_points_chart_y,
            color=colors,
            s=10
        )

    # Plot counterfactual point
    cf_pred_x_1 = np.mean([*range(len(single_points))])
    plt.scatter(
        [cf_score], 
        [cf_pred_x_1], 
        color=BLUE_TR if cf_score >= 0.5 else PINK1_TR, 
        s=100)

    # Plot a vertical line at the threshold
    plt.axvline(x=0.5, color=RED1, zorder=0, linewidth=0.5)

    # Plot a vertical line at the factual score
    plt.axvline(x=factual_score, color=PINK1_TR,
                linestyle='dashed', zorder=0)
    plt.text(factual_score - 0.06, len(text_features) *
             1.01 - 0.9, 'Factual Score', color=PINK1_TR)

    # Plot a vertical line at the counterfactual score
    plt.axvline(x=cf_score, color=BLUE_TR,
                linestyle='dashed', zorder=0)
    plt.text(cf_score - 0.10, len(text_features)*1.01 - 0.9,
             'Counterfactual Score', color=BLUE_TR)

    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Plot classes names
    class_0_name = class_names[0]
    class_1_name = class_names[1]
    size_factual_class = mpl.textpath.TextPath(
        (0, 0), class_0_name).get_extents().width * 0.002
    plt.text(0.48 - size_factual_class - max([len(f) for f in text_features])*0.0001,
             len(text_features) - 0.9 + len(text_features)*0.04, class_0_name, color=PINK1_TR, fontweight='bold')
    plt.text(0.49, len(text_features) - 0.9 + len(text_features)*0.04, '➜',
             color=RED1, fontweight='bold')
    plt.text(0.51, len(text_features) - 0.9 + len(text_features)*0.04,
             class_1_name, color=BLUE_TR, fontweight='bold')

    # Plot Counterfactual lines
    for i in range(len(single_points)):
        x_0 = point_to_pred[i]
        x_1 = cf_score
        y_0 = i
        y_1 = cf_pred_x_1
        plt.plot(
            [x_0, x_1], 
            [y_0, y_1], 
            color='k', 
            zorder=0,
            linewidth=1, 
            alpha=0.15, 
            linestyle='dotted')

    for points, x_value in mulitple_points_chart:
        for origin_point in points:
            x_0 = point_to_pred[origin_point]
            x_1 = x_value
            y_0 = origin_point
            y_1 = np.mean(points)

            plt.plot(
                [x_0, x_1], 
                [y_0, y_1], 
                color='#e0e0e0',
                zorder=0, 
                linewidth=1, 
                alpha=0.5
                )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
