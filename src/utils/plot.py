import plotly.graph_objects as go
import torch

def discrete_heatmap(time_series: torch.Tensor):
    num_time_steps, grid_x, grid_y = time_series.shape

    fig = go.Figure(
        data=[go.Heatmap(z=time_series[0], colorscale="cividis", colorbar=dict(title="Value"))],
        layout=go.Layout(
            template="simple_white",
            title=dict(
                text="Time-Evolving Heatmap",
                x=0.5,  # Center title
                font=dict(size=18, family="Arial, sans-serif")
            ),
            xaxis=dict(scaleanchor="y", visible=False),
            yaxis=dict(visible=False),

            # Add Play/Pause buttons with better positioning
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}],
                        "label": "▶ Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": "⏸ Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "showactive": True,
                "type": "buttons",
                "x": 0.15, "xanchor": "right",
                "y": -0.15, "yanchor": "bottom",
                "pad": {"r": 10, "t": 10}
            }],

            # Improved slider with better spacing and visibility
            sliders=[{
                "active": 0,
                "currentvalue": {
                    "prefix": "Time Step: ",
                    "xanchor": "right",
                    "font": {"size": 16, "color": "#333"}
                },
                "pad": {"t": 20, "b": 10},
                "len": 0.9,
                "x": 0.1, "y": -0.1,
                "steps": [
                    {
                        "args": [[str(t)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": str(t),
                        "method": "animate"
                    } for t in range(num_time_steps)
                ]
            }]
        ),
        frames=[
            go.Frame(
                data=[go.Heatmap(z=time_series[t], colorscale='Viridis')],
                name=str(t)
            ) for t in range(num_time_steps)
        ]
    )

    return fig
