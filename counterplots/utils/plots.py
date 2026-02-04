import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import plotly.graph_objects as go
from .constellation_plot_dash import ConstellationPlotDash
from ._COLORS import RED1, BLUE1, PINK1, PINK1_TR, BLUE1, BLUE_TR, LIGHT_GREY, DARK_GREY

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

def make_greedy_plot(factual_score, features_data, class_names, save_path, decision_threshold=0.5):
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
            color=PINK1_TR if point < decision_threshold else BLUE_TR,
            s=100)

    features_names = ['Factual']
    for feat_idx, f in enumerate(features_data):
        features_names.append(
            f'{f["name"]} ({f["factual"]}➜{f["counterfactual"]})')
    max_feat_names_length = max([len(feat_name) for feat_name in features_names])

    # Plot a vertical line at the threshold score
    plt.axvline(x=decision_threshold, color=RED1, zorder=0, linewidth=0.5)

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
    
    return fig, fig.gca()


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

    def create_bar(start_x, end_x, no_beak=False):
        width = end_x - start_x
        direction = 1 if width >= 0 else -1
        beak_size = min(4, abs(.2*width))
        verts = [
            (start_x, 0.),                    # inner bottom
            (start_x,                       direction * bar_y_height),  # inner top
            (end_x - direction * beak_size, direction * bar_y_height),  # before beak
            (end_x,                         direction * bar_y_height),  # beak tip (control point)
            (end_x, 0.),                    # outer bottom
            (0., 0.),                       # ignored
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
    current_x_pos = 0
    current_x_neg = 0
    x_left_pos = []
    for i, feat_data in enumerate(features_data):

        cshap_score = feat_data['x']
        if cshap_score > 0: current_x, yscale = current_x_pos, 1
        else: current_x, yscale = current_x_neg, -1

        x_left_pos.append(current_x)
        x_end = current_x + cshap_score * 100 / scale_x


        # Plot bar
        ax.add_patch(
            patches.PathPatch(
                create_bar(start_x=current_x, end_x=x_end), 
                facecolor=color_bar, 
                lw=0)
            )

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

        if cshap_score > 0: current_x_pos = x_end
        else: current_x_neg = x_end
        
        prev_score = feat_data['score']



    # Draw bar for the factual score
    plt.bar(0, scale_y - 10, width=0.5, color=PINK1, linewidth=1)
    plt.text(-5, scale_y + fontsize * 4, 'Factual Score', color=PINK1, fontsize=fontsize)
    plt.text(0, scale_y, factual_score, color=PINK1, fontsize=fontsize)

    # Plot bar for the counterfactual score
    plt.bar(current_x_pos, scale_y - 10, width=0.5, color=BLUE1, linewidth=1)
    plt.text(current_x_pos - 10, scale_y + fontsize * 4, 'Counterfactual Score', color=BLUE1, fontsize=fontsize)
    plt.text(
        current_x_pos,
        scale_y,
        round(features_data[-1]['score'], 2),
        color=BLUE1,
        fontsize=fontsize)

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
    plt.bar(current_x_pos, -scale_y + 10, width=0.2, color='#cccccc', linewidth=0)  # divider at the end

    ax.set_xlim(current_x_neg, current_x_pos+.5)
    ax.set_ylim(-scale_y, scale_y + 70)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    return fig, ax


def make_constellation_plot(
    factual_score, 
    single_points_chart, 
    text_features, 
    multiple_points_chart,
    multiple_points_chart_y, 
    single_points, 
    class_names, 
    cf_score, 
    point_to_pred, 
    save_path,
    highlight_changed_feature=None,
    decision_threshold=0.5
    ):
    """
    Create a constellation plot visualizing feature change contributions.
    
    Args:
        factual_score (float): The factual prediction score
        
        single_points_chart (np.ndarray): 2D array of shape (N, 2) mapping feature CHANGE index 
            to prediction value for single feature changes. Each row is [feature_change_idx, score].
            Example: array([[0., 0.154], [1., 0.187], [2., 0.214]])
            Used to plot the larger dots representing single feature changes.
        
        text_features (list): List of strings with feature names and their value changes.
            Format: "feature_name (factual_value➜counterfactual_value)"
            Example: ["age (25➜30)", "income (50000➜60000)"]
        
        multiple_points_chart (np.ndarray): Array of shape (M,) with dtype=object, where each 
            element is [array([feature_change_indices]), score]. Maps feature CHANGE index 
            coalitions to their prediction scores, excluding the case where ALL features are changed.
            Example: array([[array([0, 1]), 0.334], [array([0, 2]), 0.370], [array([1, 2]), 0.429]], 
            dtype=object)
            Used to plot the smaller dots representing multiple simultaneous feature changes.
        
        multiple_points_chart_y (list): List of y-axis positions for the multiple change points.
            Each value is the mean of the feature change indices in the coalition.
            Example: [0.5, 1.0, 1.5]
        
        single_points (np.ndarray): 2D array of shape (N, 2) mapping original feature index 
            (NOT feature CHANGE index) to prediction values. Format: [feature_idx, score].
            Example: array([[0., 0.154], [1., 0.187], [8., 0.214]])
            Note: The feature index in this array is not actually used in the plotting logic.
        
        class_names (dict): Mapping of class indices to class names.
            Example: {0: "Rejected", 1: "Accepted"}
        
        cf_score (float): Counterfactual prediction score
        
        point_to_pred (dict): Mapping of feature CHANGE indices to their single-change prediction scores.
            Example: {0: 0.154, 1: 0.187, 2: 0.214}
        
        save_path (str, optional): Path to save the plot. If None, plot is not saved.
        
        highlight_changed_feature (int, optional): Index of the changed feature to highlight
            (0 to N-1, where N is the number of features that differ between factual and
            counterfactual). When specified, only coalitions containing this feature change
            will be emphasized. Defaults to None (all features shown with equal emphasis).
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    if highlight_changed_feature is not None: assert isinstance(highlight_changed_feature, int), "Please pass a feature change index (int) for the keyword `highlight_changed_feature`"
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
        color = BLUE_TR if score >= decision_threshold else PINK1_TR
        if highlight_changed_feature is not None and highlight_changed_feature != adapted_feat_ind: color=LIGHT_GREY
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
    if len(multiple_points_chart) > 0:
        adapted_feat_inds_per_point = multiple_points_chart[:, 0]  # average height ~ average adapted feat inds.
        scores = multiple_points_chart[:, 1]
        if highlight_changed_feature is None:
            # No specific feature highlighted: keep original sign-based coloring
            colors = ['#E0423A' if s < decision_threshold else 'blue' for s in scores]
        else:
            # When highlighting a feature, de-emphasize all coalitions that do not contain it
            colors = [
                '#E0423A' if s < decision_threshold and highlight_changed_feature in f_ind
                else 'blue' if s >= decision_threshold and highlight_changed_feature in f_ind
                else LIGHT_GREY
                for s, f_ind in zip(scores, adapted_feat_inds_per_point)
            ]
        plt.scatter(
            scores,
            multiple_points_chart_y,
            color=colors,
            s=10
        )

    # Plot counterfactual point
    cf_pred_x_1 = np.mean([*range(len(single_points))])
    plt.scatter(
        [cf_score], 
        [cf_pred_x_1], 
        color=BLUE_TR if cf_score >= decision_threshold else PINK1_TR, 
        s=100)

    # Plot a vertical line at the threshold
    plt.axvline(x=decision_threshold, color=RED1, zorder=0, linewidth=0.5)

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
        if highlight_changed_feature is not None and i != highlight_changed_feature: continue
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

    for points, x_value in multiple_points_chart:
        for origin_point in points:
            if highlight_changed_feature is not None and origin_point != highlight_changed_feature: continue
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

    return fig, ax


def make_constellation_plot_dash(
    factual_score, 
    single_points_chart, 
    text_features, 
    multiple_points_chart,
    multiple_points_chart_y, 
    single_points, 
    class_names, 
    cf_score, 
    point_to_pred,
    highlight_feature=None,
    debug=True,
    port=8050,
    decision_threshold=0.5
):
    """
    Create and run an interactive constellation plot using Dash.
    
    Click on coalition points to highlight only the connected features and edges.
    Click the Reset button or click the same coalition again to return to normal view.
    """
    plot = ConstellationPlotDash(
        factual_score=factual_score,
        single_points_chart=single_points_chart,
        text_features=text_features,
        multiple_points_chart=multiple_points_chart,
        multiple_points_chart_y=multiple_points_chart_y,
        single_points=single_points,
        class_names=class_names,
        cf_score=cf_score,
        point_to_pred=point_to_pred,
        highlight_feature=highlight_feature,
        decision_threshold=decision_threshold
    )
    
    plot.run(debug=debug, port=port)
    return plot
