from dash import Dash, html, dcc
from rlfinitegames.dash_app.source.components import ids
import i18n
from dash.dependencies import Input, Output


def render(app: Dash) -> html.Div:
    @app.callback(
        Output(ids.GRID_WORLD_ACTION_VALUE, "children"),
        [
            Input(ids.GRID_WORLD_ACTIONS, "value"),
        ],
    )
    def update_grid_action_value(dropdown_value: int) -> str:
        # print(f"dropdown value is {dropdown_value}")
        return str(dropdown_value)

    return html.Div(
        children=[
            html.P(i18n.t("general.grid-actions")),
            dcc.Dropdown(
                id=ids.GRID_WORLD_ACTIONS,
                options=[
                    {'label': 'Down', 'value': 0},
                    {'label': 'Right', 'value': 1},
                    {'label': 'Up', 'value': 2},
                    {'label': 'Left', 'value': 3}
                ],
                value=0
            ),
            html.Div(str(0), id=ids.GRID_WORLD_ACTION_VALUE,
                     style={'display': 'none'})
        ]
    )
