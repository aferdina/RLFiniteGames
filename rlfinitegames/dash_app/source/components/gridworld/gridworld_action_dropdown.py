from dash import Dash, html, dcc
from rlfinitegames.dash_app.source.components import ids
import i18n

def render(app: Dash) -> html.Div:

    return html.Div(
        children=[
            html.H6(i18n.t("general.gridactions")),
            dcc.Dropdown(
                id=ids.GRID_WORLD_ACTIONS,
                options=[
                    {'label': 'Down', 'value': 0},
                    {'label': 'Right', 'value': 1},
                    {'label': 'Up', 'value': 2},
                    {'label': 'Left', 'value': 3}
                ],
                value='Down'
            )
        ]
    )
