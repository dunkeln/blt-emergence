import marimo

__generated_with = "0.11.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import plotly.graph_objects as go

    from src.lattice.ca import BaseModel
    from src.lattice.ca.rule_224 import Rule_224
    return BaseModel, Rule_224, go, mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Defining the data

        **First steps:** define a simple 2-D lattice and train a transfomer architecture on that

        ## Options
        + Langevin dynamics
        + continuous-time Monte Carlo ✅
        """
    )
    return


@app.cell
def _(Rule_224):
    ca = Rule_224(shape=(200, 200))
    return (ca,)


@app.cell
def _(ca, np):
    time_series = np.stack([next(ca).lattice for _ in range(100)], axis=0)
    # time_series_fat = np.repeat(time_series, 5, axis=0)
    return (time_series,)


@app.cell
def _(go, time_series):
    num_time_steps, grid_x, grid_y = time_series.shape

    fig = go.Figure(
        data=[go.Heatmap(z=time_series[0], colorscale='cividis')],
        layout=go.Layout(
            template="simple_white",
            title="Time-Evolving Heatmap",
            xaxis=dict(scaleanchor="y", visible=False),
            yaxis=dict(visible=False),
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "Time Step: "},
                "pad": {"t": 50},
                "steps": [{"args": [[str(t)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": str(t), "method": "animate"} for t in range(num_time_steps)]
            }]
        ),
        frames=[
            go.Frame(
                data=[go.Heatmap(z=time_series[t], colorscale='Viridis')],
                name=str(t)
            ) for t in range(num_time_steps)
        ]
    )

    fig
    return fig, grid_x, grid_y, num_time_steps


if __name__ == "__main__":
    app.run()
