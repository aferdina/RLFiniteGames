from dash import Dash, html
from dash import dcc
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
                Col([dcc.Markdown(i18n.t('general.grid-world-info-message'), dangerously_allow_html=True)], class_name='p-3 mt-3 me-3 bg-light'),
                Col([dcc.Markdown(i18n.t('general.grid-world-agent-info-message'), dangerously_allow_html=True)], class_name='p-3 mt-3 bg-light')
            ]),
            Row([
                Col([
                    Row([Col([
                        gridworld_action_dropdown.render(app)]),
                        # Add a button to update the plot
                        Col([Button('Update Plot', id=ids.GRID_WORLD_UPDATE_BUTTON,
                           class_name="me-2", n_clicks=None)])
                    ]),
                    Row([Col([
                        gridworld_game.render(app)],class_name='p-3 mt-3 bg-light'),
                        # Add a button to update the plot
                        Col([Textarea(id=ids.GRID_WORLD_TEXT_BOX, value=i18n.t('general.grid-world-text-box-default'))], class_name='p-3 bg-light'),
                    ])],
                    className='p-3 mt-3 mb-3 me-3 bg-light rounded-3',
                    ),
                Col([Row([
                        Col([gridworld_start_training.render(app)]),
                        Col([grid_world_play_trained_agent.render(app)]),
                     Row([grid_world_game_trained_agent.render(app)],class_name='p-3 mt-3 bg-light')
                     ])],
                    className='p-3 mt-3 mb-3 bg-light rounded-3'
                    )
            ]
            )
        ])
    )
