import numpy as np
from dash import Dash, dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from ._COLORS import RED1, BLUE1, PINK1, BLUE1, LIGHT_GREY, DARK_GREY

class ConstellationPlotDash:
    """Interactive constellation plot using Dash for full interactivity."""
    
    def __init__(
        self,
        factual_score, 
        single_points_chart, 
        text_features, 
        multiple_points_chart,
        multiple_points_chart_y, 
        single_points, 
        class_names, 
        cf_score, 
        point_to_pred,
        decision_threshold=0.5
    ):
        self.factual_score = factual_score
        self.single_points_chart = single_points_chart
        self.text_features = text_features
        self.multiple_points_chart = multiple_points_chart
        self.multiple_points_chart_y = multiple_points_chart_y
        self.single_points = single_points
        self.class_names = class_names
        self.cf_score = cf_score
        self.point_to_pred = point_to_pred
        self.decision_threshold = decision_threshold
        
        self.y_lim_low = -0.5
        self.y_lim_high = len(single_points_chart) +.5
        self.x_lim_low = -.05
        self.x_lim_high = 1.
        self.x_width = (self.x_lim_high - self.x_lim_low)
        self.cf_pred_x_1 = np.mean([*range(len(single_points))])
        
        # Store coalition information
        # First add single features as coalitions of size 1
        self.coalition_info = []
        for i in range(len(single_points_chart)):
            self.coalition_info.append([i])
        
        # Then add multi-feature coalitions
        if len(multiple_points_chart) > 0:
            for points, score in multiple_points_chart:
                self.coalition_info.append(list(map(int, points)))
        
        self.app = Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
    
    def _create_base_figure(self, selected_coalition=None):
        """Create the figure with optional coalition highlighting."""
        fig = go.Figure()
        
        # Determine which features are active
        if selected_coalition is not None:
            coalition_idx = selected_coalition
            active_features = set(self.coalition_info[coalition_idx])
        else:
            active_features = None
        
        # Add vertical lines
        fig.add_vline(x=self.decision_threshold, line_color=RED1, line_width=1, opacity=0.5)
        fig.add_vline(x=self.factual_score, line_color=PINK1, line_dash="dash", line_width=1.5)
        fig.add_vline(x=self.cf_score, line_color=BLUE1, line_dash="dash", line_width=1.5)
        
        fig.add_annotation(
            x=self.factual_score, 
            y=len(self.text_features),
            text='Factual Score',
            showarrow=False,
            font=dict(color=PINK1, size=12),
            bgcolor="#ffffff"
        )
        fig.add_annotation(
            x=self.cf_score, 
            y=len(self.text_features),
            text='Counterfactual Score',
            showarrow=False,
            font=dict(color=BLUE1, size=12),
            bgcolor="#ffffff"

        )
        
        # Class names
        fig.add_annotation(
            x=self.decision_threshold-.02*self.x_width,
            y=len(self.text_features)-.5,
            text=self.class_names[0],
            showarrow=False,
            font=dict(color=PINK1, size=12, family='Arial Black'),
            bgcolor="#ffffff"
        )
        fig.add_annotation(
            x=self.decision_threshold-.01*self.x_width,
            y=len(self.text_features)-.5,
            text='➜',
            showarrow=False,
            font=dict(color=RED1, size=12, family='Arial Black'),
            bgcolor="#ffffff"

        )
        fig.add_annotation(
            x=self.decision_threshold+.01*self.x_width,
            y=len(self.text_features)-.5,
            text=self.class_names[1],
            showarrow=False,
            font=dict(color=BLUE1, size=12, family='Arial Black'),
            bgcolor="#ffffff"
        )
        
        # Single points
        single_x = []
        single_y = []
        single_colors = []
        single_sizes = []
        
        for p in self.single_points_chart:
            adapted_feat_ind, score = p
            if active_features is not None:
                if adapted_feat_ind in active_features:
                    color = BLUE1 if score >= self.decision_threshold else PINK1
                    size = 20  # Larger when selected
                else: color, size = LIGHT_GREY, 15
            else:
                color = BLUE1 if score >= self.decision_threshold else PINK1
                size = 15
            
            single_x.append(score)
            single_y.append(adapted_feat_ind)
            single_colors.append(color)
            single_sizes.append(size)
        
        fig.add_trace(go.Scatter(
            x=single_x,
            y=single_y,
            mode='markers',
            marker=dict(size=single_sizes, color=single_colors),
            customdata=[i for i in range(len(self.single_points_chart))],
            hovertemplate='Score: %{x:.3f}<extra></extra>',
            name='single_features',
            showlegend=False
        ))
        
        # Lines from single points to counterfactual (only show when no selection)
        if active_features is None:
            for i in range(len(self.single_points)):
                x_0 = self.point_to_pred[i]
                x_1 = self.cf_score
                y_0 = i
                y_1 = self.cf_pred_x_1
                
                fig.add_trace(go.Scatter(
                    x=[x_0, x_1],
                    y=[y_0, y_1],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0.15)', width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Counterfactual point
        cf_color = BLUE1 if self.cf_score >= self.decision_threshold else PINK1
        fig.add_trace(go.Scatter(
            x=[self.cf_score],
            y=[self.cf_pred_x_1],
            mode='markers',
            marker=dict(size=15, color=cf_color),
            hovertemplate='Counterfactual<br>Score: %{x:.3f}<extra></extra>',
            name='Counterfactual',
            showlegend=False
        ))
        
        # Coalition connections
        if len(self.multiple_points_chart) > 0:
            # Coalition lines
            for coalition_idx, ((points, x_value), y_val) in enumerate(
                zip(self.multiple_points_chart, self.multiple_points_chart_y)
            ):
                # Only show lines if this coalition is selected or no selection
                if selected_coalition is not None:
                    adjusted_coalition_idx = selected_coalition - len(self.single_points)
                    if coalition_idx != adjusted_coalition_idx:
                        continue
                
                for origin_point in points:
                    x_0 = self.point_to_pred[origin_point]
                    x_1 = x_value
                    y_0 = origin_point
                    y_1 = y_val
                    
                    fig.add_trace(go.Scatter(
                        x=[x_0, x_1],
                        y=[y_0, y_1],
                        mode='lines',
                        line=dict(color=LIGHT_GREY, width=.5),
                        showlegend=False,
                        hoverinfo='skip',
                        zorder=-1
                    ))
            
            # Coalition points
            adapted_feat_inds_per_point = self.multiple_points_chart[:, 0]
            scores = self.multiple_points_chart[:, 1]
            
            multi_x = []
            multi_y = []
            multi_colors = []
            multi_sizes = []
            multi_customdata = []
            multi_text = []
            
            for i, (s, f_ind, y_val) in enumerate(
                zip(scores, adapted_feat_inds_per_point, self.multiple_points_chart_y)
            ):
                if selected_coalition is not None:
                    # Check if this is the selected multi-feature coalition
                    adjusted_coalition_idx = selected_coalition - len(self.single_points)
                    if i == adjusted_coalition_idx:
                        color = '#E0423A' if s < self.decision_threshold else 'blue'
                        size = 10
                    else:
                        color = LIGHT_GREY
                        size = 5
                else:
                    if s < self.decision_threshold: color = '#E0423A'
                    else: color = 'blue'
                    size = 5

                multi_x.append(s)
                multi_y.append(y_val)
                multi_colors.append(color)
                multi_sizes.append(size)
                multi_customdata.append(i + len(self.single_points_chart))  # offset by amount of size-1 coalitions
                features_str = [
                    "• " + self.text_features[int(f)]
                    for f in sorted(f_ind)
                    ]
                multi_text.append('<br>'.join(features_str[::-1]))

            fig.add_trace(go.Scatter(
                x=multi_x,
                y=multi_y,
                mode='markers',
                marker=dict(size=multi_sizes, color=multi_colors),
                customdata=multi_customdata,
                hovertemplate='%{text}<br>Score: %{x:.3f}<extra></extra>',
                text=multi_text,
                name='Coalitions',
                showlegend=False
            ))

        # Layout
        fig.update_layout(
            width=1000,
            height=400,
            xaxis=dict(
                range=[self.x_lim_low, self.x_lim_high],
                title='Score',
                showgrid=True,
                gridcolor='rgba(200,200,200,0.2)'
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(self.text_features))),
                ticktext=self.text_features
            ),
            plot_bgcolor='white',
            hovermode='closest',
            margin=dict(l=200, r=50, t=50, b=50),
            clickmode='event+select'
        )
        
        return fig
    
    def _setup_layout(self):
        """Setup the Dash app layout."""
        self.app.layout = html.Div([
            html.H2("Interactive Constellation Plot", style={'textAlign': 'center'}),
            html.Div([
                html.Button('Reset Selection', id='reset-button', n_clicks=0,
                           style={'marginBottom': '10px'})
            ], style={'textAlign': 'center'}),
            dcc.Graph(
                id='constellation-plot',
                figure=self._create_base_figure(),
                config={'displayModeBar': True}
            ),
            html.Div(id='selected-coalition', style={'textAlign': 'center', 'marginTop': '10px'}),
            # Hidden div to store selected coalition
            html.Div(id='selected-coalition-store', style={'display': 'none'})
        ])
    
    def _setup_callbacks(self):
        """Setup Dash callbacks for interactivity."""
        @self.app.callback(
            [Output('constellation-plot', 'figure'),
             Output('selected-coalition', 'children'),
             Output('selected-coalition-store', 'children')],
            [Input('constellation-plot', 'clickData'),
             Input('reset-button', 'n_clicks')],
            [State('selected-coalition-store', 'children')]
        )
        def update_plot(clickData, reset_clicks, current_selection):
            ctx = callback_context
            
            if not ctx.triggered:
                return self._create_base_figure(), "", None
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Reset button clicked
            if trigger_id == 'reset-button':
                return self._create_base_figure(), "", None
            
            # Plot clicked
            if trigger_id == 'constellation-plot' and clickData:
                point = clickData['points'][0]
                
                if 'customdata' in point:
                    selected_coalition_idx = int(point['customdata'])                    
                    # Toggle selection
                    if current_selection == selected_coalition_idx:
                        # Deselect
                        fig = self._create_base_figure()
                        message = ""
                        selected_coalition_idx = None
                    else:
                        # Select coalition
                        fig = self._create_base_figure(selected_coalition=selected_coalition_idx, is_single=False)
                        features_str = ', '.join(map(str, self.coalition_info[selected_coalition_idx]))
                        message = f"Selected Coalition: Features [{features_str}]"
                    
                    return fig, message, selected_coalition_idx
        
            # No change
            return self._create_base_figure(), "", None
    
    def run(self, debug=True, port=8050):
        """Run the Dash app."""
        self.app.run(debug=debug, port=port)
