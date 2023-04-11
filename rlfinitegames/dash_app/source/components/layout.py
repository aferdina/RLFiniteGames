from dash import Dash, html
from dash_bootstrap_components import Container, Row, Col, Button
from rlfinitegames.dash_app.source.components.gridworld import gridworld_action_dropdown, gridworld_game, gridworld_start_training, grid_world_play_trained_agent, grid_world_game_trained_agent
from dash.dcc import Textarea
import i18n
from rlfinitegames.dash_app.source.components import ids


def create_layout(app: Dash) -> html.Div:
    return html.Div(
        className="app-div",
        children=Container([
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
                Col([gridworld_start_training.render(app),
                     grid_world_play_trained_agent.render(app),
                     grid_world_game_trained_agent.render(app)
                     ]
                    )
            ]
            )
        ])
    )
