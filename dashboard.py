import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
from flask import Flask, Response

import time
import json
import pandas as pd
import numpy as np

from Pose_Estimation_Modul import SkipCounter
import utils


server = Flask(__name__)
app = Dash(external_stylesheets=[dbc.themes.CYBORG], server=server)


light_blue = [51, 58, 84]
blue = [54, 115, 234]

################ Figure Methods #################

def create_skipping_time_fig(break_time, skipping_time):
    fig = px.bar(
        data_frame=pd.DataFrame(
            {
                "holder": [0, 0],
                "time": [skipping_time, break_time],
                "col": ["skip", "break"],
            }
        ),
        x="holder",
        y="time",
        labels=False,
        text=[
            utils.format_seconds(skipping_time, "Skipping"),
            utils.format_seconds(break_time, "Break"),
        ],
        height=420,
        color="col",
        color_discrete_sequence=[
            "rgba(242, 190, 0, 1.)",
            "rgba(242, 190, 0, .5)",
        ],
        facet_col_spacing=0,
        range_x=[-0.3, 0.3],
        range_y=[0, break_time + skipping_time],
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(
            l=0,  # left
            r=0,  # right
            t=0,  # top
            b=0,  # bottom
        ),
        yaxis_title=None,
        xaxis_title=None,
        xaxis_visible=False,
        yaxis_visible=False,
        plot_bgcolor="rgba(0,0,0, 0.0)",
        paper_bgcolor="rgba(0,0,0, 0.0)",
    )
    fig.update_traces(
        textfont_size=16,
        textangle=0,
        textposition="inside",
        cliponaxis=False,
        marker_line_width=0,
        textfont_color="white",
    )

    return fig


def create_skipping_speed_graph(speed_list):
    fig = px.line(
        x=speed_list[:, 0],
        y=speed_list[:, 1],
        line_shape="spline",
        height=120,
    )
    fig.update_traces(line_color="#3673EA", line_width=3)
    fig.update_layout(
        margin=dict(
            l=0,  # left
            r=0,  # right
            t=0,  # top
            b=0,  # bottom
        ),
        plot_bgcolor="rgba(0,0,0, 0.0)",
        paper_bgcolor="rgba(0,0,0, 0.0)",
    )
    max_speed = np.asarray(speed_list)[:, 1].max()
    if max_speed == 0:
        fig.update_layout(yaxis_range=[0, 1])
    fig.update_yaxes(
        title=None,
        color="white",
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(255,255,255,.2)",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="rgba(255,255,255,.5)",
    )
    # create ticks for x-axis
    n_ticks = 5
    vals = speed_list[:, 0]
    if len(vals) > (n_ticks - 1):
        idx = np.round(np.linspace(0, len(vals) - 1, n_ticks)).astype(int)
        current_x_tickvals = vals[idx]
    else:
        current_x_tickvals = vals

    fig.update_xaxes(
        color="white",
        showgrid=False,
        zeroline=False,
        title=None,
        tickmode="array",
        tickvals=current_x_tickvals,
        ticktext=[utils.seconds_to_min(x) for x in current_x_tickvals],
    )
    return fig

def return_skipping_indicator(is_skipping):
    color = "green" if is_skipping else "red"
    text = "Your are \n skipping" if is_skipping else "Your are \n NOT skipping"
    return html.Div(
        id="is_skipping",
        className="is_skipping",
        style={"background-color": color},
        children=[text],
    )


################ Video Stream #################

def gen():
    while True:
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + cap.get_frame() + b"\r\n\r\n"
        )

@server.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


################ Dashboard Layout #################

time_col = [
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(
                        className="time_div",
                        children=[
                            html.Div(
                                className="time_title",
                                children=["Total Session Time"],
                            ),
                            html.Div(
                                id="total_time",
                                className="time",
                                children=["00:00min"],
                            ),
                        ],
                    ),
                ]
            )
        ]
    ),
    dbc.Row(
        [
            dcc.Graph(
                id="time_graph",
                className="time_graph",
                figure=create_skipping_time_fig(0, 0),
            )
        ]
    ),
]

video_col = [
    dbc.Row([html.Img(src="/video_feed", width="600px")]),
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(
                        className="speed_graph_div",
                        children=[
                            "Speed: Skips per Minute",
                            dcc.Graph(
                                id="speed_graph",
                                figure=create_skipping_speed_graph(np.array([[0, 0]])),
                            ),
                        ],
                    )
                ],
            )
        ]
    ),
]

skipping_indicator_row = dbc.Row(
    [
        dbc.Col(
            [
                html.Div(
                    id="is_skipping_holder",
                    children=[
                        html.Div(
                            id="is_skipping",
                            className="is_skipping",
                            style={"background-color": "red"},
                        )
                    ],
                )
            ]
        ),
        dbc.Col(
            [
                dbc.Row(
                    [
                        html.Button(
                            children="Reset",
                            id="reset_btn",
                            n_clicks=0,
                            style={"width": "192px"},
                        )
                    ]
                ),
                dbc.Row(
                    [
                        html.Button(
                            children="Take a break",
                            id="break_btn",
                            n_clicks=0,
                            style={"width": "192px"},
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        html.Button(
                            children="Quit and Save",
                            id="save_btn",
                            n_clicks=0,
                            style={"width": "192px"},
                        ),
                    ]
                ),
            ],
        ),
    ]
)


skip_row = dbc.Row(
    [
        dbc.Col(
            [
                html.Div(
                    className="tile counter_holder",
                    children=[
                        html.Div(
                            className="skip_title",
                            children=["Total Skips"],
                        ),
                        html.Div(
                            id="counter",
                            className="counter",
                            children=[0],
                        ),
                    ],
                ),
            ]
        ),
    ]
)

speed_row = dbc.Row(
    [
        dbc.Col(
            [
                html.Div(
                    className="tile speed_holder",
                    id="speed_holder",
                    style={"background-color": "#333A54"},
                    children=[
                        html.Div(
                            className="skip_title",
                            children=["Current Speed"],
                        ),
                        html.Div(
                            id="speed_indicator",
                            className="counter",
                            children=[0],
                        ),
                    ],
                ),
            ]
        ),
        dbc.Col(
            [
                html.Div(
                    className="tile max_speed_holder",
                    children=[
                        html.Div(
                            className="skip_title",
                            children=["Max. Speed"],
                        ),
                        html.Div(
                            id="max_speed_indicator",
                            className="counter",
                            children=[0],
                        ),
                    ],
                ),
            ]
        ),
    ]
)

app.layout = app.layout = dbc.Container(
    [
        html.H3("AI Skipping Counter"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    time_col,
                    width=2,
                ),
                dbc.Col(
                    video_col,
                    width=6,
                ),
                dbc.Col(
                    [
                        skipping_indicator_row,
                        skip_row,
                        speed_row,
                        html.Div(id="fps"),# style={"display": "none"}),
                    ]
                ),
            ]
        ),
        dcc.Interval(
            id="interval_component",
            interval=1000,
            n_intervals=0,
            disabled=True, 
        ),
    ],
)

################ Dashboard Callback Methods #################

@app.callback(
    [
        Output("counter", "children"),
        Output("total_time", "children"),
        Output("speed_graph", "figure"),
        Output("is_skipping_holder", "children"),
        Output("time_graph", "figure"),
        Output("speed_indicator", "children"),
        Output("fps", "children"),
        Output("max_speed_indicator", "children"),
        Output("speed_holder", "style"),
    ],
    Input("interval_component", "n_intervals"),
    prevent_initial_call=True,
)
def update_dashboard_metrics(n):
    (
        count,
        total_time,
        skipping_time,
        break_time,
        speed,
        speed_list,
        max_speed,
        is_skipping,
        fps_str,
    ) = cap.get_data()

    speed_holder_style = {
        "background-color": utils.get_cmap_color([light_blue, blue], max_speed, speed)
    }

    return (
        count,
        utils.format_seconds(total_time),
        create_skipping_speed_graph(speed_list),
        return_skipping_indicator(is_skipping),
        create_skipping_time_fig(break_time, skipping_time),
        speed,
        fps_str,
        max_speed,
        speed_holder_style,
    )


@app.callback(
    [Output("interval_component", "disabled"), Output("break_btn", "children")],
    Input("break_btn", "n_clicks"),
    State("interval_component", "disabled"),
)
def paus_session(n, disabled):
    if disabled:
        return False, "Take a Break"
    else:
        cap.break_time = time.time()
        return True, "Resume"


@app.callback(
    Output("interval_component", "n_intervals"),
    [
        Input("reset_btn", "n_clicks"),
        Input("save_btn", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def save_session(reset_n, save_n):
    if dash.callback_context.triggered_id == "save_btn":
        # save session
        (
            count,
            total_time,
            skipping_time,
            break_time,
            speed,
            speed_list,
            max_speed,
            is_skipping,
            fps_str,
        ) = cap.get_data()

        save_json = dict(
            count=count,
            total_time=total_time,
            skipping_time=skipping_time,
            break_time=break_time,
            max_speed=max_speed,
        )
        with open("skip_sessions/skip.json", "w") as fp:
            json.dump(save_json, fp)

    # reset session
    cap.init_specs()
    return 0


if __name__ == "__main__":
    cap = SkipCounter()
    app.run_server(debug=True)
