from dash import Dash, html
from dash_bootstrap_components import Container, Row, Col, Button
from rlfinitegames.dash_app.source.components.gridworld import gridworld_action_dropdown, gridworld_game
from dash.dcc import Textarea, Dropdown
import i18n
from rlfinitegames.dash_app.source.components import ids

def create_layout(app: Dash) -> html.Div:
    return html.Div(
        className="app-div",
        children= Container([
    Row([
        Col([
            # Add a button to update the plot
            gridworld_action_dropdown.render(app),
            # Add a button to update the plot
            Button('Update Plot', id=ids.GRID_WORLD_UPDATE_BUTTON,
                       class_name="me-2", n_clicks=None),
            # Add game environment to the game plot
            gridworld_game.render(app),
            Textarea(id=ids.GRID_WORLD_TEXT_BOX, value=i18n.t('general.grid-world-text-box-default'))]),
        Col([Dropdown(
            id='mouse-button2',
            options=[
                {'label': 'Down', 'value': 0},
                {'label': 'Right', 'value': 1},
                {'label': 'Up', 'value': 2},
                {'label': 'Left', 'value': 3}
            ],
            value='Down'
        ),Button('Update Plot', id='update-button2',
                       class_name="me-2", n_clicks=0),
            Textarea(id='text-output2', value="Let the game started")]
        )
    ]
    )
])
    )