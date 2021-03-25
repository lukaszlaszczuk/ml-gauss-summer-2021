import numpy as np
import pandas as pd
import plotly.graph_objects as go


class PlotUtil:
    
    def __init(self):
        pass
    
    def plot_gradient_descent(df, x_col, y_col, thetas_snapshot, title_value, x_min, x_max, y_min, y_max, title_label='alpha'):
        x_test = np.linspace(x_min, x_max, 1000)
        fig_dict = {
            "data": [],
            "layout": {},
            "frames": []
        }

        fig_dict["layout"]["xaxis"] = {"range": [x_min, x_max], "title": x_col}
        fig_dict["layout"]["yaxis"] = {"range": [y_min, y_max],"title": y_col}
        fig_dict["layout"]["hovermode"] = "closest"
        fig_dict["layout"]["updatemenus"] = [
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": False},
                                        "fromcurrent": True, "transition": {"duration": 300,
                                                                            "easing": "quadratic-in-out"}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
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
            }
        ]

        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "n_iter, loss_value, theta_0, theta_1:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": []
        }


        frame = {'data': [], 'name': str(thetas_snapshot[1][1])}
        data_dict = {
            'x': x_test,
            'y': thetas_snapshot[0][0][0]+thetas_snapshot[0][0][1]*x_test,
            'mode': 'markers'
        }
        fig_dict['data'].append(data_dict)

        data_dict = {
            'x': df[x_col],
            'y': df[y_col],
            'mode': 'markers'
        }
        fig_dict['data'].append(data_dict)


        for theta in thetas_snapshot:
            frame = {'data': [], 'name': str(theta[1])}
        #     print(theta[0][0]+theta[0][1]*x_test)
            data_dict = {
                'x': df[x_col],
                'y': df[y_col],
                'mode': 'markers'
            }
            frame['data'].append(data_dict)
            frame = {'data': [], 'name': str(theta[1])}
            data_dict = {
                'x': x_test,
                'y': theta[0][0]+theta[0][1]*x_test,
                'mode': 'markers'
            }
            frame['data'].append(data_dict)

            fig_dict['frames'].append(frame)
            slider_step = {'args': [
                [theta[1]],
                {'frame': {'duration': 300, 'redraw': False},
                 'mode': 'immediate',
               'transition': {'duration': 300}}
             ],
             'label': f"<br>{theta[1]}, {PlotUtil._loss_function(theta[0][0]+theta[0][1]*df[x_col], df[y_col])}, {theta[0][0]}, {theta[0][1]}",
             'method': 'animate'}
            sliders_dict['steps'].append(slider_step)

        fig_dict["layout"]["sliders"] = [sliders_dict]
        fig = go.Figure(fig_dict)
        fig.update_layout(title=f"Gradient descent for {title_label} = {title_value}")
        return fig
    
    @staticmethod
    def _loss_function(y_pred, y_real):
        return 1/2 * np.mean((y_pred-y_real)**2)
    